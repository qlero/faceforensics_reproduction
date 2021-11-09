"""
    ---- Overview ----
    Implements the functions necessary to run an xception
    pre-trained binary classification model to evaluates 
    a folder of video files (formatted as .mp4).
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
    def __init__(self, num_class=2):
        """
        Initializes the class.
        ---
        parameters:
            num_class    : number of output classes.
        """
        super(TransferModel, self).__init__()
        # Imports the model structure
        self.model = xception_model()
        # replaces the last FC layer with a 2-class one for transfer
        # learning purposes
        feature_number = self.model.last_linear.in_features
        self.model.last_linear = nn.Linear(feature_number, num_class)
    def forward(self, x):
        x = self.model(x)
        return x

# Function declarations

def model_selection(num_classes):
    """
    Generates the transfer learning model.
    ---
    parameter:
        num_class  : number of output classes.
    """
    model = TransferModel(
            num_class=num_classes
            )
    return model, 299, True, ["image"], None

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
