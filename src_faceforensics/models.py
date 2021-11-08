"""
    ---- Overview ----
    Implements the functions necessary to run an xception or 
    resnet pre-trained binary classification model to evaluates 
    a folder of video files (formatted as .mp4).
    Code adapted by A. Rossler from Francois Chollet.
    ------------------
"""

# Library Imports

import argparse
import math
import os
import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision

from torch.nn import init

# Global variable declarations

mod = "xception-b5690688"
xception_url = f"http://data.lip6.fr/cadene/pretrainedmodels/{mod}.pth"
pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': xception_url,
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the 
                            # validation transform should 
                            # be 333, and make sure to center 
                            # crop at 299x299
        }
    }
}

# Class declaration

class Block(nn.Module):
    """
    Declares a PyTorch block object.
    """
    def __init__(
            self,
            in_filters, out_filters, reps,
            strides=1, 
            start_with_relu=True, grow_first=True
        ):
        super(Block, self).__init__()
        # Computes and creates the different layers of a given block
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(
                    in_filters, out_filters,
                    1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        self.relu = nn.ReLU(inplace=True)
        rep=[]
        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(
                in_filters, out_filters,
                3, stride=1, padding=1,
                bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
            for _ in range(reps-1):
                rep.append(self.relu)
                rep.append(SeparableConv2d(
                    filters, filters,
                    3, stride=1, padding=1,
                    bias=False))
                rep.append(nn.BatchNorm2d(filters))
        else:
            for _ in range(reps-1):
                rep.append(self.relu)
                rep.append(SeparableConv2d(
                    filters, filters,
                    3, stride=1, padding=1,
                    bias=False))
                rep.append(nn.BatchNorm2d(filters))
            rep.append(self.relu)
            rep.append(SeparableConv2d(
                in_filters, out_filters,
                3, stride=1, padding=1,
                bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)
    def forward(self,inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x+=skip
        return x

class SeparableConv2d(nn.Module):
    """
    Declares a separable convolutional 2-dimensional layer in PyTorch.
    """
    def __init__(
            self,
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, dilation=1,
            bias=False
        ):
        super(SeparableConv2d,self).__init__()
        self.conv1 = nn.Conv2d(
                in_channels, in_channels,
                kernel_size, stride, padding, dilation,
                groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
                in_channels,
                out_channels,
                1,1,0,1,1,
                bias=bias)
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class TransferModel(nn.Module):
    """
    Transfer learning model that takes an imagenet pre-trained model
    with a fully connected layer as a base model.
    """
    def __init__(self, model_choice, num_class=2):
        """
        Initializes the class.
        ---
        parameters:
            model_choice : name of the model to use (covered: xception,
                            resnet50, resnet18). 
            num_class    : number of output classes.
        """
        super(TransferModel, self).__init__()
        self.model_choice = model_choice
        # Imports the model structure
        if model_choice == "xception":
            self.model = xception_model()
        elif model_choice == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)
        elif model_choice == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=True)
        else:
            raise Exception(f"Model {model_choice} not implemented.")
        # replaces the last FC layer with a 2-class one for transfer
        # learning purposes
        feature_number = self.model.last_linear.in_features
        self.model.last_linear = nn.Linear(feature_number, num_class)

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, specified in:
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ 
        Constructor function.
        ---
        parameters:
            num_classes : number of classes for the output later
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1=Block(
                64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(
                128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(
                256,728,2,2,start_with_relu=True,grow_first=True)
        self.block4=Block(
                728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(
                728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(
                728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(
                728,728,3,1,start_with_relu=True,grow_first=True)
        self.block8=Block(
                728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(
                728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(
                728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(
                728,728,3,1,start_with_relu=True,grow_first=True)
        self.block12=Block(
                728,1024,2,2,start_with_relu=True,grow_first=False)
        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, num_classes)
    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x
    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

# Function declarations

def model_selection(model_choice, num_classes):
    """
    Generates the transfer learning model.
    ---
    parameters:
        model_choice : name of the model to use (covered: xception,
                            resnet50, resnet18). 
        num_class    : number of output classes.
    """
    model = TransferModel(
            model_choice=model_choice,
            num_classes=num_classes
            )
    if model_choice == "xception":
        return model, 299, True, ["image"], None
    elif model_choice == "resnet18" or model_choice == "resnet50":
        return model, 224, True, ["image"], None
    else:
        raise NotImplementedError(model_choice)

def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        check = setting["num_classes"]
        assert num_classes == check, \
            f"num_classes should be {check}, but is {num_classes}"
        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    model.last_linear = model.fc
    del model.fc
    return model

def xception_model():
    """
    Imports an Xception model and sets its values to a pre-trained
    set of weights stored locally.
    """
    model = xception()
    model.fc = model.last_linear
    del model.last_linear
    # Imports the pre-trained weights
    state_dict = torch.load("./input_models/xception-b5690688.pth")
    for name, weights in state_dict.items():
        if "pointwise" in name: 
            state_dict[name] = weigths.unsqueeze(-1).unsqueeze(-1)
    model.load_state_dict(state_dict)
    # Cleans the last linear layer
    model.last_linear = model.fc
    del model.fc
    return model
