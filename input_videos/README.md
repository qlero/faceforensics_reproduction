# Content

This folder contains the videos in ``.mp4`` used to perform tests of the pre-trained models.

### download.py script

You will not find the download script for the original dataset in this repository as part of the license of FaceForensics++ dataset. Head over to their [github](https://github.com/ondyari/FaceForensics/tree/master/dataset) to request for access via their google form.

### convert_videos2images.py script

This script converts all the videos downloaded via the ``download.py`` script into series of images cropped at the face bounding box.

### Note on downloading all videos

This repository is aimed at self-hosted, local testing and thus will focus on the c40 videos. To obtain the c40 videos via the ``download.py`` script, you can run the following script:

```sh
python download.py . -d all -c c40 -t videos --server EU2
```
