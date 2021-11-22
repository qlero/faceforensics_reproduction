"""
    ----- Overview -----
    Implements a resnet with the following example as base:
    https://marekpaulik.medium.com/imbalanced-dataset-image-classification-with-pytorch-6de864982eb1
    --------------------
"""

# Library Imports

import os
import pandas as pd
import torch
import torchvision

from collections import Counter
from tqdm import tqdm

from torch.nn import CrossEntropyLoss, Linear
from torch.optim import Adam
from torchvision.models import resnet50
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

# Config declaration

cfg = {
    "batch_size": 128,
    "data_dir": "dataloader_tree_c40" ,
    "num_classes": 2,
    "device": torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu"),
    "sample": True,
    "loss_weights": False,
    "tensorboard": True,
    "stop_early": True,
    "patience": 5,
    "use_amp": True,
    "freeze_backbone": True,
    "unfreeze_after": 5,
    "num_epochs": 25,
    "lr": 1.5e-05,
    "seed": 0
}

# Class Declarations

class EarlyStopping:
    """
    Early stopping callback declaration
    """
    def __init__(self, patience=1, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.path= path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score:
            self.counter +=1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class ResnetClassifier():
    """
    Declares a ResNet Classifier.
    """
    def __init__(
        self, num_classes, device,
        sample=cfg["sample"], 
        loss_weights=cfg["loss_weights"], 
        batch_size=cfg["batch_size"],
        lr=cfg["lr"], 
        tensorboard=cfg["tensorboard"], 
        stop_early=cfg["stop_early"], 
        use_amp=cfg["use_amp"], 
        freeze_backbone=cfg["freeze_backbone"]
    ):
        """
        Initializes the Classifier.
        ---
        parameters:
            sample          : True if data is imbalanced, RWS will be used.
            loss_weights    : True if data is imbalanced, weight params will
                              be passed to the loss function.
            freeze_backbone : If pretrained architecture, freezes it
        """
        self.num_classes = num_classes
        self.device = device
        self.sample = sample
        self.loss_weights = loss_weights
        self.batch_size = batch_size
        self.lr = lr
        self.tensorboard = tensorboard
        self.stop_early = stop_early
        self.use_amp = use_amp
        self.freeze_backbone = freeze_backbone

    def load_model(self):
        """
        Loads the pre-trained resnet 50 architecture.
        """
        # Loads the architecture
        self.model = resnet50(pretrained=True)
        # Freeze the layers except for the output
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = Linear(
            in_features=self.model.fc.in_features, 
            out_features=self.num_classes
        )
        self.model = self.model.to(self.device)
        # Sets the optimizer: Adam
        self.optimizer = Adam(self.model.parameters(), self.lr) 
        # Sets the class weights if indicated
        if self.loss_weights:
            class_count = Counter(self.train_classes)
            class_weights = torch.Tensor([
                len(self.train_classes)/c 
                for c in pd.Series(class_count).sort_index().values
            ]) # Cant iterate over class_count because dictionary is unordered
            class_weights = class_weights.to(self.device)  
            self.criterion = CrossEntropyLoss(class_weights)
        else:
            self.criterion = CrossEntropyLoss() 

    def fit_one_epoch(self, train_loader, scaler, epoch, num_epochs):
        """
        Fits the network for one epoch.
        ---
        parameters:
            train_loader : Data loader to fit over.
            scaler       : Scaler for automatic mixed precision (AMP)
                           https://pytorch.org/docs/stable/amp.html
            epoch        : Epoch number
            num_epochs   : Total number of epochs
        """
        # Declares variables
        step_train = 0
        train_losses = list() # Every epoch check average loss per batch 
        train_acc = list()
        # Sets the model for training + a loading bar for looking at
        # batch processing
        self.model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        # Iterates over each batch
        for i, (images, targets) in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp): #mixed precision
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            # Prepares the gradient descent
            self.optimizer.zero_grad()
            train_losses.append(loss.item())
            # Calculates running train accuracy
            predictions = torch.argmax(logits, dim=1)
            num_correct = sum(predictions.eq(targets))
            running_train_acc = float(num_correct) / float(images.shape[0])
            train_acc.append(running_train_acc)
            # Plot to tensorboard
            if self.tensorboard:
                img_grid = make_grid(images[:10])
                self.writer.add_image("Cassava_images", img_grid) 
                # Check how transformed images look in training
                self.writer.add_scalar(
                    "training_loss", 
                    loss, 
                    global_step=step_train
                )
                self.writer.add_scalar(
                    "training_acc", 
                    running_train_acc, 
                    global_step=step_train
                )
                step_train +=1
        train_loss = torch.tensor(train_losses).mean()
        # prints results.
        print(f"Epoch {epoch}/{num_epochs-1}")  
        print(f"Training loss: {train_loss:.2f}")

    def val_one_epoch(self, val_loader, scaler):
        """
        Performs a validation sep at the end of one fit epoch.
        ---
        parameters:
            val_loader : Data loader to validate over
            scaler     : Scaler for automatic mixed precision
        """
        # Declares variables
        val_losses = list()
        val_accs = list()
        self.model.eval()
        step_val = 0
        # Iterates over each batch of the validation loader
        with torch.no_grad():
            for (images, targets) in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                val_losses.append(loss.item())      
                # Performs predictions and computes accuracy
                predictions = torch.argmax(logits, dim=1)
                num_correct = sum(predictions.eq(targets))
                running_val_acc = float(num_correct) / float(images.shape[0])
                val_accs.append(running_val_acc)
                if self.tensorboard:
                    self.writer.add_scalar(
                        "validation_loss", 
                        loss, 
                        global_step=step_val
                    )
                    self.writer.add_scalar(
                        "validation_acc", 
                        running_val_acc, 
                        global_step=step_val
                    )
                    step_val +=1
            # Records the results and prints
            self.val_loss = torch.tensor(val_losses).mean()
            val_acc = torch.tensor(val_accs).mean() # Average acc per batch
            print(f"Validation loss: {self.val_loss:.2f}")  
            print(f"Validation accuracy: {val_acc:.2f}") 

    def fit(
        self, train_loader, val_loader, 
        num_epochs=10, unfreeze_after=5, 
        checkpoint_dir="checkpoint.pt"
    ):
        """
        Fits the network.
        ---
        parameters:
            train_loader   : Training data loader
            val_loader     : Validation data loader
            num_epochs     : Total number of epochs
            unfreeze_after : Nb of epochs after which to unfreeze the 
                             resnet parameters
            checkpoint_dir : name of the file checkpoint
        """
        # Declares the tensorboard object and callback settings
        if self.tensorboard:
            self.writer = SummaryWriter("sampler") 
        if self.stop_early:
            early_stopping = EarlyStopping(
            patience=cfg["patience"], 
            path=checkpoint_dir)
        # Declares the automatic mixed precision object
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        # Fit over the number of epochs
        for epoch in range(num_epochs):
            if self.freeze_backbone:
                if epoch == unfreeze_after:  # Unfreeze after x epochs
                    for param in self.model.parameters():
                        param.requires_grad = True
            self.fit_one_epoch(train_loader, scaler, epoch, num_epochs)
            self.val_one_epoch(val_loader, scaler)
            if self.stop_early:
                early_stopping(self.val_loss, self.model)
                if early_stopping.early_stop:
                    print("Early Stopping")
                    print(f"Best validation loss: {early_stopping.best_score}")
                    break


