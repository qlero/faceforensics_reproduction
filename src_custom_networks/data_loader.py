"""
    ---- Overview ----
    Script to create a data loader out of a given folder of images.
    This folder of images is supposed to have been build using the
    script create_data_folder.py locaded in the input_videos folder.
    ------------------
"""

# Library Imports

import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Resize

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
        return image, label

# Function declarations

def create_dataloader(
        target_folder, labels_csv, 
        batch_size=64, resize_to=[200, 200]
    ):
    """
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
            labels_csv, target_folder, transform=Resize(resize_to)
        )
    # To visualize a picture do:
    # train_features, train_labels = next(iter(dataloader))
    # img = train_features[0].squeeze()
    # plt.imshow(img[0,:,:], cmap="gray")
    # plt.show()
    return DataLoader(dataloader, batch_size=batch_size, shuffle=True)
