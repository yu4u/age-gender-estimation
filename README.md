# Age and Gender Estimation
This is a Keras implementation of a CNN for estimating age and gender from a face image [1, 2].
In training, [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) is used.

- [Aug. 21, 2020] Refactored; use tensorflow.keras
- [Jun. 30, 2019] [Another PyTorch-based project](https://github.com/yu4u/age-estimation-pytorch) was released
- [Nov. 12, 2018] Enable Adam optimizer; seems to be better than momentum SGD
- [Sep. 23, 2018] Demo from directory
- [Aug. 11, 2018] Add age estimation sub-project [here](age_estimation)
- [Jul. 5, 2018] The UTKFace dataset became available for training.
- [Apr. 10, 2018] Evaluation result on the APPA-REAL dataset was added.

## Dependencies
- Python3.6+

Tested on:
- Ubuntu 16.04, Python 3.6.9, Tensorflow 2.3.0, CUDA 10.01, cuDNN 7.6


## Usage

### Use trained model for demo
Run the demo script (requires web cam).
You can use `--image_dir [IMAGE_DIR]` option to use images in the `[IMAGE_DIR]` directory instead.

```sh
python demo.py
```

The trained model will be automatically downloaded to the `pretrained_models` directory.

### Create training data from the IMDB-WIKI dataset
First, download the dataset.
The dataset is downloaded and extracted to the `data` directory by:

```sh
./download.sh
```

Secondly, filter out noise data and serialize labels into `.csv` file.
Please check [check_dataset.ipynb](check_dataset.ipynb) for the details of the dataset.
The training data is created by:

```sh
python create_db.py --db imdb
```

```sh
usage: create_db.py [-h] [--db DB] [--min_score MIN_SCORE]

This script cleans-up noisy labels and creates database for training.

optional arguments:
  -h, --help            show this help message and exit
  --db DB               dataset; wiki or imdb (default: imdb)
  --min_score MIN_SCORE minimum face_score (default: 1.0)
```

The resulting files with default parameters are included in this repo (meta/imdb.csv and meta/wiki.csv),
thus there is no need to run this by yourself.


### Create training data from the UTKFace dataset [currently not supported]
Firstly, download images from [the website of the UTKFace dataset](https://susanqq.github.io/UTKFace/).
`UTKFace.tar.gz` can be downloaded from `Aligned&Cropped Faces` in Datasets section.
Then, extract the archive.

```sh
tar zxf UTKFace.tar.gz UTKFace
```

Finally, run the following script to create the training data:

```
python create_db_utkface.py -i UTKFace -o UTKFace.mat
```

[NOTE]: Because the face images in the UTKFace dataset is tightly cropped (there is no margin around the face region),
faces should also be cropped in `demo.py` if weights trained by the UTKFace dataset is used.
Please set the margin argument to 0 for tight cropping:

```sh
python demo.py --weight_file WEIGHT_FILE --margin 0
```

The pre-trained weights can be found [here](https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.29-3.76_utk.hdf5).

### Train model
Train the model architecture using the training data created above:

```sh
python train.py
```

Trained weight files are stored as `checkpoints/*.hdf5` for each epoch if the validation loss becomes minimum over previous epochs.

#### Changing model or the other training parameters
You can change [default setting(s)](src/config.yaml) from command line as:

```sh
python train.py model.model_name=EfficientNetB3 model.batch_size=64
```

Available models can be found [here](https://keras.io/api/applications/).

#### Check training curve
The training logs can be easily visualized via [wandb](https://www.wandb.com/) by:

1. create account from [here](https://app.wandb.ai/login?signup=true)
2. create new project in wandb (e.g. "age-gender-estimation")
3. run `wandb login` on terminal and authorize
4. run training script with `wandb.project=age-gender-estimation` argument
5. check dashboard!

### Use the trained model

```sh
python demo.py
```

```sh
usage: demo.py [-h] [--weight_file WEIGHT_FILE] [--margin MARGIN]
               [--image_dir IMAGE_DIR]

This script detects faces from web cam input, and estimates age and gender for
the detected faces.

optional arguments:
  -h, --help            show this help message and exit
  --weight_file WEIGHT_FILE
                        path to weight file (e.g. weights.28-3.73.hdf5)
                        (default: None)
  --margin MARGIN       margin around detected face for age-gender estimation
                        (default: 0.4)
  --image_dir IMAGE_DIR
                        target image directory; if set, images in image_dir
                        are used instead of webcam (default: None)
```

Please use the best model among `checkpoints/*.hdf5` for `WEIGHT_FILE` if you use your own trained models.



### Estimated results
Trained on imdb, tested on wiki.
![](https://github.com/yu4u/age-gender-estimation/wiki/images/result.png)


### Evaluation

#### Evaluation on the APPA-REAL dataset
You can evaluate a trained model on the APPA-REAL (validation) dataset by:

```bash
python evaluate_appa_real.py --weight_file WEIGHT_FILE
```

Please refer to [here](appa-real) for the details of the APPA-REAL dataset.

The results of trained model is:

```
MAE Apparent: 5.33
MAE Real: 6.22
```

The best result reported in [5] is:

```
MAE Apparent: 4.08
MAE Real: 5.30
```

Please note that the above result was achieved by finetuning the model using the training set of the APPA-REAL dataset.

## License
This project is released under the MIT license.
However, [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) used in this project is originally provided under the following conditions.

> Please notice that this dataset is made available for academic research purpose only. All the images are collected from the Internet, and the copyright belongs to the original owners. If any of the images belongs to you and you would like it removed, please kindly inform us, we will remove it from our dataset immediately.

Therefore, the pretrained model(s) included in this repository is restricted by these conditions (available for academic research purpose only).


## References
[1] R. Rothe, R. Timofte, and L. V. Gool, "DEX: Deep EXpectation of apparent age from a single image," in Proc. of ICCV, 2015.

[2] R. Rothe, R. Timofte, and L. V. Gool, "Deep expectation of real and apparent age from a single image
without facial landmarks," in IJCV, 2016.

[3] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," in arXiv:1710.09412, 2017.

[4] Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random Erasing Data Augmentation," in arXiv:1708.04896, 2017.

[5] E. Agustsson, R. Timofte, S. Escalera, X. Baro, I. Guyon, and R. Rothe, "Apparent and real age estimation in still images with deep residual regressors on APPA-REAL database," in Proc. of FG, 2017.
