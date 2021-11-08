# Content

This is a Python 3.8.x reimplementation of the testing part of the FaceForensics++ code available [here](https://github.com/ondyari/FaceForensics), kindly provided by [Andreas Rössler](https://github.com/ondyari).

This repository also contains an overview presentation of the FaceForensics++ paper based on [Alejandro Serrano](https://github.com/serras)'s [INFOFP 2021 presentation](https://github.com/serras/infofp-2021). 

### Installation

This repository was tested with **Python 3.8.8**. The repository versions used are stored in the ``requirements.txt`` file in the root.

### Running the process

1. Download the models at the link [here](http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip) and extracts the models ``full_c23.p``, ``full_c40.p`` and ``full_raw.p`` in the compressed folder ``faceforensics++_models/faceforensics++_models_subset/full/xception`` into the folder ``input_models``.

2. Set the videos you want to test in the ``input_videos`` folder (an example in 360p taken from [Youtube](https://youtu.be/oxXpB9pSETo) is left there as an example). 

3. Run the following code depending on the model you want to run.

```sh
python src_faceforensics/launch_detection.py -i input_videos -m input_models/<model_file> -o output_videos
```
