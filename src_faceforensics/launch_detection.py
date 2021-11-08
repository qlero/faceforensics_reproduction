"""
    ---- Overview ----
    Launch the detection script using a pre-trained Xception network.
    ------------------
"""
import argparse
import os
import detect_from_videos as dfv

if __name__ == "__main__":
    # Parses the command line input
    p = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    p.add_argument("--video_path", "-i", type=str)
    p.add_argument("--model_path", "-m", type=str, default=None)
    p.add_argument("--output_path", "-o", type=str, default=".")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=None)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()
    # Checks if the input is a video or a folder
    if args.video_path[-4:] in [".mp4", ".avi"]:
        videos = [args.video_path]
    else:
        video_list = [
                v for v in os.listdir(args.video_path)
                if v[-4:] in [".mp4", ".avi"]
                ]
        videos = [os.path.join(args.video_path, v) for v in video_list]
    for video in videos:
        dfv.predict_video(
                video, args.model_path, args.output_path,
                model_name = "xception", model_as_dict = False,
                start_frame = 0, end_frame = None,
                cuda = True
                ) 
