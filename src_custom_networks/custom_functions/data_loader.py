"""
    ---- Overview ----
    Script to create a data loader out of a given folder of images.
    This folder of images is supposed to have been build using the
    script create_data_folder.py locaded in the input_videos folder.
    ------------------
"""

# Library Imports

import math
import os
import pandas as pd
import torch

from collections import Counter

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from torchvision.transforms import RandomRotation, RandomHorizontalFlip

# Class declarations

class CustomImageDataset(Dataset):
    def __init__(
            self, 
            annotations_file, img_dir, 
            transform=None, target_transform=None
        ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform: image = self.transform(image)
        if self.target_transform: label = self.target_transform(label)
        return image.float(), label

class CustomTestLoader(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = os.listdir(self.img_dir)  
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.images[idx]))
        return self.transform(img), self.images[idx] 
    
# Function declarations

def create_dataloader_v2(
        target_folder, labels_csv, 
        batch_size=64, resize_to=[200, 200]
    ):
    """
    Using a 80%/20& train/val split, construct two dataloaders with
    a Imbalance Weighted Random Sampler.
    ---
    parameters:
        target_folder : Target folder where the data is located
        labels_csv    : Path of csv file of the data labels
        batch_size    : Declares a batch size for the data loader
        resize_to     : Transforms the data images to a given size
    """
    # Declares a transform pipeline for the images
    Transform = Compose([
        ToTensor(),
        Resize(resize_to),
        RandomRotation(20),
        RandomHorizontalFlip(p=0.5),
        Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])
    # Retrieves the data stored locally
    train_full = ImageFolder(target_folder, transform=Transform)
    train_set, val_set = random_split(
            train_full, 
            [math.floor(len(train_full)*0.8), 
                math.ceil(len(train_full)*0.2)]
        )
    train_classes = [label for _, label in train_set]
    # Computes the class proportions for the WRS sampler
    class_count = Counter(train_classes)
    class_weights = torch.Tensor(
            [len(train_classes)/c 
                for c in pd.Series(class_count).sort_index().values]
            ) # Cant iterate over class_count because dictionary is unordered
    sample_weights = [0] * len(train_set)
    for idx, (image, label) in enumerate(train_set):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    # Declares the sampler to deal with Imbalance
    sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples = len(train_set), replacement=True
        )  
    # Declares the loaders and return them
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader

def create_dataloader(
        target_folder, labels_csv, 
        batch_size=64, resize_to=[200, 200]
    ):
    """
    Using a custom image dataset class, construct a dataloader with a
    given batch size and resizing transform pipeline. See:
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    https://pytorch.org/vision/stable/transforms.html
    ---
    parameters:
        target_folder : Target folder where the data is located
        labels_csv    : Path of csv file of the data labels
        batch_size    : Declares a batch size for the data loader
        resize_to     : Transforms the data images to a given size
    """
    dataloader = CustomImageDataset(
            labels_csv, target_folder, 
            transform=Compose([
                ToTensor(),
                Resize(resize_to), 
                Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
                ])
        )
    return DataLoader(dataloader, batch_size=batch_size)
