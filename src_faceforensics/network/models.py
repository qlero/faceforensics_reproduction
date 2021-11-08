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

from network.xception import xception
from torch.nn import init

# Class declaration

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
            # replaces the last FC layer with a 2-class one for transfer
            # learning purposes
            feature_number = self.model.last_linear.in_features
            self.model.last_linear = nn.Linear(feature_number, num_class)
        elif model_choice == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=True)
            feature_number = self.model.fc.in_features
            self.model.fc = nn.Linear(feature_number, num_class)
        elif model_choice == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=True)
            feature_number = self.model.fc.in_features
            self.model.fc = nn.Linear(feature_number, num_class)
        else:
            raise Exception(f"Model {model_choice} not implemented.")
    def forward(self, x):
        x = self.model(x)
        return x

# Function declarations

def model_selection(model_name, num_classes):
    """
    Generates the transfer learning model.
    ---
    parameters:
        model_name : name of the model to use (covered: xception,
                            resnet50, resnet18). 
        num_class  : number of output classes.
    """
    model = TransferModel(
            model_choice=model_name,
            num_class=num_classes
            )
    if model_name == "xception":
        return model, 299, True, ["image"], None
    elif model_name == "resnet18" or model_name == "resnet50":
        return model, 224, True, ["image"], None
    else:
        raise NotImplementedError(model_name)

def xception_model():
    """
    Imports an Xception model and sets its values to a pre-trained
    set of weights stored locally.
    """
    model = xception(pretrained=False)
    model.fc = model.last_linear
    del model.last_linear
    # Imports the pre-trained weights
    state_dict = torch.load("./input_models/xception-b5690688.pth")
    for name, weights in state_dict.items():
    	if 'pointwise' in name:
    		state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    model.load_state_dict(state_dict)
    # Cleans the last linear layer
    model.last_linear = model.fc
    del model.fc
    return model
