"""
    ---- Overview ----
    Given an xception or resnet pre-trained binary classification
    model, evaluates a folder of video files (formatted as .mp4).
    ------------------
"""

# Library Imports

import argparse
import cv2
import dlib
import os
import torch
import torch.nn as nn

from .models import model_selection
from os.path import join
from PIL import Image as pil_image
from torchvision import transforms
from tqdm import tqdm

# Global variable declarations

xception_default_data_transforms = {}
for s in ["train", "val", "test"]:
    xception_default_data_transforms[s] = transforms.Compose(
            [transforms.Resize((299, 299)),
             transforms.Resize((299, 299)),
             transforms.ToTensor()])

# Function declarations

def get_bounding_box(face, width, height, scale=1.3, min_size=None):
    """
    Generates a quadratic bounding box using the dlib library.
    --- 
    parameters:
        face    : dlib face class
        height  : frame height
        minsize : minimum b-box size
        scale   : b-box size multiplier to increase the face region
    outputs:
        x       : top left corner x coordinate 
        y       : top left corner y coordinate
        size_bb : length of sides of the square b-box
    """
    # Computes the size of the bounding box
    size_bb = int(max(
            face.right()-face.left,face.bottom()-face.top()*scale
            ))
    if minsize and size_bb < minsize:
        size_bb = minsize
    # Computes the bounding box's top-left corner
    # with an out of bound check
    x = face.left() + face.right()
    y = face.top() + face.bottom()
    x = max(int((x - size_bb) // 2), 0)
    y = max(int((y - size_bb) // 2), 0)
    # Computes the final bounding box size
    size_bb = min(size_bb, width - x, height - y)
    return x, y, size_bb

def preprocess_image(image, cuda=True):
    """
    Pre-process an image into a network input using the PIL library.
    ---
    parameters:
        image : numpy array of the input image
        cuda  : bool whether to set the image as cuda-cast
    output:
        prep_image : pytorch tensor of the input image
    """
    # Transforms image from BGR to RBG
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Casts the image to PIL format
    prep_pipeline = xception_default_data_transforms["test"]
    prep_image = prep_pipeline(pil_image.fromarray(image))
    # Adds a first dimension (the networks expects a batch)
    prep_image = prep_image.unsqueeze(0)
    if cuda: prep_image = prep_image.cuda()
    return prep_image

def predict(image, model, activation=nn.Softmax(dim=1), cuda=True):
    """
    Predicts a label given an input image, cast to cuda if
    required.
    ---
    parameters:
        image      : numpy array of the input image
        model      : pytorch model with a final linear layer
        activation : activation function
        cuda       : enables cuda (must be equivalent to model's)
    outputs:
        prediction : binary value of the prediction
        output     : output of the network
    """
    # Preprocesses the input image
    prep_image = preprocess_image(image, cuda)
    # Predicts
    output = activation(model(prep_image))
    # Casts to desired shape
    _, prediction = torch.max(output, 1)
    prediction = float(prediction.cpu().numpy())
    return int(prediction), output

def predict_video(
        video_path, model_path, output_path,
        model_name = "xception",
        start_frame=0, end_frame=None, cuda=True
        ):
    """
    ---
    parameters:
        video_path  : Path to the video to test
        model_path  : Path to model to use
        output_path : Path to the output folder to store results
        model_name  : Name of the model to use (covered: xception,
                      resnet50, resnet18)
        start_frame : Indicates where to start in the video
        end_frame   : Indicates where to stop in the video
        cuda        : Enables cuda (must be equivalent to model's)
    """
    print(f"Launching process on video: {video_path}")
    # Initiates some intermediary variables
    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    os.makedirs(output_path, exist_ok=True)
    writer = None
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1
    # Initiates a frame by frame reader object
    reader = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    # Initiates the dlib face detector
    face_detector = dlib.get_frontal_face_detector()
    # Loads a model
    model, *_ = model_selection(
            model_name=model_name, 
            n_output_classes=2
            )
    if model_path is not None:
        state_dictionary = torch.load(model_path)
        model.load_state_dict(state_dictionary, False)
        print(f"Model found in {model_path}")
    else:
        print(f"Model not found, intializing random model")
    if cuda: model.cuda()
    # Initiates frame numbers to process the tested video
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    # Initiates the progress bar
    pbar = tqdm(total=end_frame-start_frame)
    ############################
    # PERFORMS THE READER LOOP #
    # OVER THE VIDEO FRAMES    #
    ############################
    wile reader.isOpened():
        # Retrieves the next frame of the video
        pbar.update(1)
        frame_number += 1
        _, image = reader.read()
        if image is None: break
        #if frame_number < start_frame:
        #    continue
        # Retrieves the image size
        height, width = image.shape[:2]
        # Intializes the output writer object
        if writer is None:
            writer = cv2.VideoWriter(
                    join(output_path, video_fn),
                    fourcc,
                    fps,
                    (height, width)[::-1])
        # Performs the face detection with dlib
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # Retrieves the first/largest face
            face = faces[0]
            # Crops the face out of the frame picture
            x, y, size = get_bounding_box(face, width, height)
            cropped_face = image[y:y+size, x:x+size]
            ####################
            # model prediction #
            # on cropped face  #
            ####################
            prediction, output = predict_with_model(
                    cropped_face, model, cuda=cuda
                    )
            ####################
            # Construct the bounding box (with its label) to be cast
            # to the reconstructed video
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = "fake" if prediction == 1 else "real"
            color = (0,255,0) if prediction == 0 else (0,0,255)
            bb_scores = str([
                    "{0:.2f}".format(float(x)) for x in
                    output.detach().cpu().numpy()[0]
                    ]) + label
            cv2.putText(
                    image, bb_scores, (x,y+h+30), 
                    font_face, font_scale, color, thickness, 2
                    ) 
            # Cast the bounding box to the image
            cv2.rectangle(image, (x,y), (x+w,y+h), color, 2)
        # Checks if at the end of the video
        if frame_number >= end_frame:
            break
        # Shows the result in an external window
        cv2.imshow("test", image)
        cv2.waitKey(33) # restricts the showing at about 30 fps
        writer.write(image)
    # Closes the loading bar
    pbar.close()
    # Closes the writer object at end of function
    if writer is not None:
        writer.release()
        print(f"Finished! Output saved under {output_path}.")
    else:
        print("Input video file was empty.")

