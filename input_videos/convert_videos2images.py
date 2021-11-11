"""
    ---- Overview ----
    Given that the script download.py downloaded videos from the FF++
    dataset, converts those videos into images (frame by frame) for
    machine learning purposes. Images are cropped using dlib around the 
    face area.
    ------------------
"""

# Library Imports

import argparse
import cv2
import dlib
import os
import random

from PIL import Image as pil_image

# Function declarations

def get_bounding_box(face, width, height, scale=1.3, minsize=None):
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
    size_bb = max(face.right()-face.left(),face.bottom()-face.top())
    size_bb = int(size_bb*scale)
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

def convert_video2image(
        video_path, image_path,
        start_frame=0, end_frame=None
        ):
    """
    Converts a video path into a list of images.
    9 images are produced per videos, randomly selected 
    without replacement.
    ---
    parameters:
        video_path  : Path of the video to convert
        image_path  : Path where the images will be saved
        start_frame : Indicates where to start in the video
        end_frame   : Indicates where to stop in the video
    """
    print(f"Launching process on video: {video_path}")
    # Initiates a frame by frame reader object
    reader = cv2.VideoCapture(video_path)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initiates the dlib face detector
    face_detector = dlib.get_frontal_face_detector()
    # Initiates frame numbers to process the tested video
    frame_number = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    frames_to_extract = sorted(random.sample(range(end_frame), 9))
    last_frame = frames_to_extract[-1]
    ############################
    # PERFORMS THE READER LOOP #
    # OVER THE VIDEO FRAMES    #
    ############################
    while reader.isOpened():
        # Retrieves the next frame of the video
        frame_number += 1
        _, image = reader.read()
        if frame_number not in frames_to_extract: 
            if frame_number > last_frame: break
            else: continue
        if image is None: break
        #if frame_number < start_frame:
        #    continue
        # Retrieves the image size
        height, width = image.shape[:2]
        # Performs the face detection with dlib
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(grayscale_image, 1)
        if len(faces):
            # Retrieves the first/largest face
            face = faces[0]
            # Crops the face out of the frame picture
            x, y, size = get_bounding_box(face, width, height)
            cropped_face = image[y:y+size, x:x+size]
            # splits the path to retrieve a list with the 
            # video name and its extension. Splices in the frame
            # number
            name = video_path.split("/")[-1].split(".")
            name = name[:-1] + [str(frame_number)]
            name = "_".join(name)+".jpg"
            # Saves the picture
            im = pil_image.fromarray(cv2.cvtColor(
                cropped_face, cv2.COLOR_BGR2RGB
                ))            
            im.save(image_path+f"{name}")
        # Checks if at the end of the video
        if frame_number >= end_frame:
            break

def pre_process_videos(compression="c40"):
    """
    Pre-processes all the videos downloaded with the download.py
    script into images
    ---
    parameters:
        compression : compression level to be retrieved
    """
    # retrieves all the videos existing in the tree structure
    # under video_path
    fake_types = [
            "FaceSwap", "DeepFakeDetection", "Deepfakes",
            "NeuralTextures", "Face2Face", "FaceShifter"
            ]
    real_types = ["youtube", "actors"]
    fake_path = lambda t, c: f"manipulated_sequences/{t}/{c}/"
    real_path = lambda t, c: f"original_sequences/{t}/{c}/"
    cases = [(fake_types, fake_path), (real_types, real_path)]
    # Pre-process each video
    for case in cases:
        video_types = case[0]
        path_f = case[1]
        for t in video_types:
            folder = path_f(t, compression)
            image_path = folder+"images/"
            os.mkdir(image_path)
            video_list = os.listdir(folder+"videos/")
            for video in video_list:
                video_path = folder+"videos/"
                video_path = os.path.join(
                        video_path,
                        video
                        )
                convert_video2image(video_path, image_path)


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
        pre_process_videos(args.compression)
