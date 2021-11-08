"""
    ---- Overview ----
    Implements the functions necessary to run an xception or 
    resnet pre-trained binary classification model to evaluates 
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
import torchvision

from .xception import xception

# Class declaration

class TransferModel(nn.Module):
    """
    Transfer learning model that takes an imagenet pre-trained model
    with a fully connected layer as a base model.
    """
    def __init__(self, model_choice, num_out_class=2):
        super(TransferModel, self).__init__()
        self.model_choice = model_choice

# Function declarations


