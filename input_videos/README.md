# Content

This folder contains the videos in ``.mp4`` used to perform tests of the pre-trained models.

### ``download.py`` script

You will not find the download script for the original dataset in this repository as part of the license of FaceForensics++ dataset. Head over to their [github](https://github.com/ondyari/FaceForensics/tree/master/dataset) to request for access via their google form.

### ``convert_videos2images.py`` script

This script converts all the videos downloaded via the ``download.py`` script into series of images cropped at the face bounding box. Refer back to the root ``README.md`` for the description of the process.

### ``create_data_folder.py`` script

This script can create two types of data folders for import by a DataLoader function. 

The first type relies on the ``retrieve_images`` function it contains and will retrieve the converted images, split between three folders: ``train``, ``validation``, and ``test``.

The second relies on the ``retrieve_images_v2`` function and will retrieve the converted images, split between a fake and real folders (i.e. by class). 

Calling the script via the following will call the first version. 

```sh
python create_data_folder.py --compression <compression_mode>
```

Adding an argument ``--retrieve`` followed by any letter will call the second version (the check is done on whether the argument is empty, i.e. ``None``).

## Note on downloading all videos

This repository is aimed at self-hosted, local testing and thus will focus on the c40 videos. To obtain the c40 videos via the ``download.py`` script, you can run the following script:

```sh
python download.py . -d all -c c40 -t videos --server EU2
```
