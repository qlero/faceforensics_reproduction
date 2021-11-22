"""
    ----- Overview ----
    Runs a scattering resnet on the data.
    -------------------
"""
# Library imports

import argparse
import pandas as pd
import torch

from torchvision.models import resnet50
from torch.nn import Linear
from torch.utils.data import DataLoader
from torchsummary import summary

from custom_functions import data_loader as DL
from custom_functions import resnet as R

# Functions declarations

def run_inference(test_dir, device, Transform=None, n_classes = 2):
    """
    Runs an inference step on a test dataset.
    """
    # Loads the model architecture
    model = resnet50()
    # Updates the last layer to the right number of classes
    model.fc = Linear(
        in_features = model.fc.in_features, 
        out_features = n_classes
    )
    model = model.to(device)
    # Loads pre-trained parameters
    checkpoint = torch.load("checkpoint.pt")
    model.load_state_dict(checkpoint)
    model.eval()
    # Creates the test dataloader
    test_set = DL.CustomTestLoader(test_dir, transform=Transform)
    test_loader = DataLoader(test_set, batch_size=4)
    # Creates variables to store test results
    sub = pd.DataFrame(columns=["category", "id"])
    id_list = []
    pred_list = []
    with torch.no_grad():
        for (image, image_id) in test_loader:
            image = image.to(device)
            logits = model(image)
            predictions = list(torch.argmax(logits,1).cpu().numpy())
            for i in image_id:
                id_list.append(i)
            for pred in predictions:
                pred_list.append(pred)
    sub["category"] = pred_list
    sub["id"] = id_list
    sub.to_csv("predictions.csv")

if __name__ == "__main__":
    """
    Trains a simple Hybrid Resnet Scattering + CNN Model on
    a subset of the FF++ dataset. The scattering layers are 
    of the 2nd order kind.
    """
    parser = argparse.ArgumentParser(
            description = "FaceForensics++ Hybrid Scatter.Resnet"
            )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_data_folder", type=str, default=None)
    parser.add_argument("--val_data_folder", type=str, default=None)
    parser.add_argument("--train_label_csv", type=str, default=None)
    parser.add_argument("--val_label_csv", type=str, default=None)
    parser.add_argument("--resize_to", nargs="*", type=int, default=[200,200])
    parser.add_argument("--width_resnet", type=int, default=2)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--weighted_sampler", type=bool, default=False)
    args = parser.parse_args()

    # clears the torch cuda cache
    torch.cuda.empty_cache()

    # Checks for args
    if args.weighted_sampler:
        if args.train_data_folder is None:
            raise ValueError("Image folder(s)/Label file(s) must be properly set")
    elif not args.weighted_sampler:
        if args.train_data_folder is None or args.train_label_csv is None \
                or args.val_data_folder is None or args.val_label_csv is None:
            raise ValueError("Image folder(s)/Label file(s) must be properly set")

    # Checks if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Declares the model
    print("STEP 1: Declares the model")
    model = R.ResnetClassifier(
            num_classes = 2,
            device = device
        )
    model.load_model()
    # Declares the dataloaders
    print("STEP 2: Declares the data loaders")
    if args.weighted_sampler:
        train_loader, val_loader = DL.create_dataloader_v2(
            args.train_data_folder, args.train_label_csv,
            args.batch_size, args.resize_to
        )
    else:
        train_loader = DL.create_dataloader(
            args.train_data_folder, args.train_label_csv,
            args.batch_size, args.resize_to
        )
        val_loader = DL.create_dataloader(
            args.val_data_folder, args.val_label_csv,
            args.batch_size, args.resize_to
        )
    # Optimizer
    print("STEP 3: Running the learning process")
    model.fit(
            train_loader, val_loader,
            num_epochs = args.n_epochs,
            unfreeze_after=5
        )



