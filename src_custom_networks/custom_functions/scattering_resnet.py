"""
    ----- Overview -----
    Implements an hybrid scattering resnet with the library Kymatio with
    the following example as base:
    https://www.kymat.io/gallery_2d/cifar_resnet_torch.html
    --------------------
"""

# Library Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from torchvision import datasets, transforms

# Class Declarations

class BasicBlock(nn.Module):
    """
    Declares a basic convolutional block for a neural network.
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Scattering2dResNet(nn.Module):
    """
    Declares a Scattering ResNet block from the Kymatio library.
    """
    def __init__(self, in_channels, k=2, n=4, num_classes=1):
        super(Scattering2dResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.K = in_channels
        self.init_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
            nn.Conv2d(in_channels, self.ichannels,
                  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.ichannels),
            nn.ReLU(True)
        )
        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(x.size(0), self.K, 50, 50)
        x = self.init_conv(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Function Declarations

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(F.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return correct_results_sum, acc

def conv3x3(in_planes, out_planes, stride=1):
    """
    Implements a 3 by 3 convolution layer with padding.
    ---
    parameters:
        in_planes  : Indicates the input shape
        out_planes : Indicates the output shape
        stride     : Indicates the convolution stride
    """
    layer =  nn.Conv2d(
            in_planes, out_planes, 
            kernel_size=3, stride=stride, 
            padding=1, bias=False
        )
    return layer

def train(model, device, train_loader, optimizer, epoch, scattering):
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            output = model(scattering(data)).squeeze()
            loss = nn.BCEWithLogitsLoss()(output, torch.Tensor.float(target))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def val(model, device, val_loader, scattering):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(scattering(data)).squeeze()
            val_loss += nn.BCEWithLogitsLoss()(
                    output, torch.Tensor.float(target)
                ).item() # sum up batch loss
            corr, batch_acc = binary_acc(output, target)
            correct += corr
    print('\nVal set: total loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
