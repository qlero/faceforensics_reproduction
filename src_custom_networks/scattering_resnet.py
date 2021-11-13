"""
    ----- Overview ----
    Runs a scattering resnet on the data.
    -------------------
"""
# Library imports

import argparse
import torch

from kymatio.torch import Scattering2D
from custom_functions import data_loader as DL
from custom_functions import scattering_resnet as SR
from torchsummary import summary

# Functions declarations

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
    
    # Clears the torch cuda cache
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
    # Declares the Scattering Block format
    print("STEP 1: Declares the invariant scattering layer")
    scattering = Scattering2D(J=2, shape=args.resize_to)
    K = 81*3
    if use_cuda:
        scattering = scattering.cuda()
    # Declares the model
    print("STEP 2: Declares the model")
    model = SR.Scattering2dResNet(K, args.width_resnet).to(device)
    summary(model, (3, 81, 50, 50))
    # Declares the dataloaders
    print("STEP 3: Declares the data loaders")
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
    learning_rate = 1e-6
    print("STEP 4: Running the learning process")
    for epoch in range(args.n_epochs):
        # reduces the learning rate every twenty step, starting at 0
        if epoch % 20 == 0:
            optimizer = torch.optim.Adam(
                    model.parameters(), lr=learning_rate, 
                )
            learning_rate *= 0.5
        SR.train(model, device, train_loader, optimizer, epoch+1, scattering)
        SR.val(model, device, val_loader, scattering)

