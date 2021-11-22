# Content

This folder contains the implementation of custom neural networks (i.e. not using pre-trained models by the FaceForensics++ team).

### Networks available

The networks available are a Scattering Resnet and a Resnet that can respectively be called with the script:

- ``scattering_resnet.py``
- ``resnet.py``

#### Script without Weighted Random Sampling

This requires to run the `create_data_folder.py` script first in the `input_videos` folder.

```sh
python <model_file.py> \
	--batch_size 64 \
	--train_data_folder '../input_videos/dataloader_c40/train_validation/' \
	--test_data_folder '../input_videos/dataloader_c40/test/' \
	--train_label_csv '../input_videos/train_metadata.csv' \
	--test_label_csv '../input_videos/test_metadata.csv' \
	--resize_to 200 200 \
	--width_resnet 5 \
	--n_epochs 50
```

#### Script with Weighted Random Sampling

This requires to run the `create_data_folder.py` script first in the `input_videos` folder, with the `--retrieve` keyword.

```sh
python <model_file.py>.py \
	--batch_size 64 \
	--train_data_folder '../input_videos/dataloader_tree_c40/' \
	--resize_to 200 200 \
	--width_resnet 1 \
	--n_epochs 10 \
	--weighted_sampler 1
```
