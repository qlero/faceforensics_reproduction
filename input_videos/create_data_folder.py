"""
    ---- Overview ----
    Given that the script convert_videos2images.py was run and exported
    a select few frames from downloaded videos, moves those images into
    two folders (train/val and test) at the same level as this script to
    create custom dataloaders with PyTorch.
    ------------------
"""

# Library Imports

import argparse
import csv
import os
import random
import shutil

# Function declarations

def retrieve_images(compression="c40"):
    """
    Retrieves the images that were retrieved from the videos and
    moves them to a dataloader folder.
    ---
    parameters:
        compression : compression level to be retrieved
    """
    fake_types = [
            "Deepfakes", "FaceSwap", "DeepFakeDetection",
            "NeuralTextures", "Face2Face", "FaceShifter"
            ]
    real_types = ["youtube", "actors"]
    fake_path = lambda t, c: f"manipulated_sequences/{t}/{c}/images/"
    real_path = lambda t, c: f"original_sequences/{t}/{c}/images/"
    cases = [(fake_types, fake_path), (real_types, real_path)]
    real_images = []
    fake_images = []
    # Retrieves the path of each image, the image name, and its label
    for label, case in enumerate(cases):
        video_types = case[0]
        path_f = case[1]
        for t in video_types:
            path = path_f(t, compression)
            images = os.listdir(path)
            paths = [path]*len(images)
            labels = [label]*len(images)
            concat = list(zip(paths, images, labels))
            if label==0: fake_images += concat
            else: real_images += concat
    random.shuffle(real_images)
    random.shuffle(fake_images)
    # creates a folder for the data loader and splits the data between
    # a train/val and test subfolders
    new_path = f"dataloader_{compression}/"
    new_tr_path = new_path + "train_validation/"
    new_te_path = new_path + "test/"
    os.mkdir(new_path)
    os.mkdir(new_tr_path)
    os.mkdir(new_te_path)
    # shaves the list of images to retrieve to have an exact 80k number
    nb_real = len(real_images)
    nb_fake = len(fake_images)
    nb_elements = nb_real + nb_fake
    if nb_elements > 80000:
        nb_to_shave = nb_elements - 80000
        end_nb_real = len(real_images) // 1000 * 1000
        real_images = real_images[:end_nb_real]
        fake_images = fake_images[:-(nb_to_shave - nb_real + end_nb_real)]
        image_list = real_images + fake_images
    random.shuffle(image_list)
    # Splits the list into train/val and test sets with 18.75% i.e. 15k images
    # going into test
    split_element = 80000 - 15000
    train_list = image_list[:split_element]
    test_list = image_list[split_element:]
    # Moves the train/val images to a new folder
    for image in train_list:
        src = image[0] + image[1]
        tar = new_tr_path + image[1]
        shutil.copyfile(src, tar)
    for image in test_list:
        src = image[0] + image[1]
        tar = new_te_path + image[1]
        shutil.copyfile(src, tar)
    # Creates the CSV data then files
    csv_train = [image_metadata[1:] for image_metadata in train_list]
    csv_test = [image_metadata[1:] for image_metadata in test_list]
    for data in [("train", csv_train), ("test", csv_test)]:
        with open(f"{data[0]}_metadata.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "label"])
            for row in data[1]:
                writer.writerow(row)

if __name__ == "__main__":
    # Parses the command line input
    p = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    p.add_argument("--compression", "-c", type=str)
    args = p.parse_args()
    # Checks if the compression input is valid
    if args.compression is None:
        print("Please enter a compression argument.")
    elif args.compression not in ["raw", "c23", "c40"]:
        print(f"Compression argument '{args.compression}', not valid")
    else:
        retrieve_images(args.compression)
