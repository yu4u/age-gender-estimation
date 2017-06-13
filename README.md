# Age and Gender Estimation
This is a Keras implementation of a CNN network for estimating age and gender from a face image [1, 2].
In training, [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) is used.


## Dependencies
- Python3.5+
- Keras2.0+
- scipy, numpy, Pandas, tqdm, tables, h5py
- dlib (for demo)
- OpenCV3

Tested on:
- Ubuntu 16.04, Python 3.5.2, Keras 2.0.3, Tensorflow(-gpu) 1.0.1, CUDA 8.0, cuDNN 5.0
- macOS Sierra, Python 3.6.0, Keras 2.0.2, Tensorflow 1.0.0


## Usage

### Use pretrained model
Download pretrained model

```sh
mkdir -p pretrained_models
wget -P pretrained_models https://www.dropbox.com/s/rf8hgoev8uqjv3z/weights.18-4.06.hdf5
```

Run demo script (requires web cam)

```sh
python3 demo.py
```

### Train a model using the IMDB-WIKI dataset

#### Download the dataset
The dataset is downloaded and extracted to the `data` directory.

```sh
./download.sh
```

#### Create data
Filter out noise data and serialize images and labels for training into `.mat` file.
Please check [check_dataset.ipynb](check_dataset.ipynb) for the details of the dataset.
```sh
python3 create_db.py --output data/imdb_db.mat --db imdb --img_size 64
```

#### Train network
Train the network using the training data created above.

```sh
python3 train.py --input data/imdb_db.mat
```

Plot training curves from history file.

```sh
python3 plot_history.py -input models/history_16_8.h5 
```

![](https://github.com/yu4u/age-gender-estimation/wiki/images/loss.png)

![](https://github.com/yu4u/age-gender-estimation/wiki/images/accuracy.png)

## Network architecture
In [the original paper](https://www.vision.ee.ethz.ch/en/publications/papers/articles/eth_biwi_01299.pdf) [1, 2], the pretrained VGG network is adopted.
Here the Wide Residual Network (WideResNet) is trained from scratch.
I modified the @asmith26's implementation of the WideResNet; two classification layers (for age and gender estimation) are added on the top of the WideResNet.

Note that while age and gender are independently estimated by different two CNNs in [1, 2], in my implementation, they are simultaneously estimated using a single CNN.


## Results
Trained on imdb, tested on wiki.
![](https://github.com/yu4u/age-gender-estimation/wiki/images/result.png)


## References
[1] R. Rothe, R. Timofte, and L. V. Gool, "DEX: Deep EXpectation of apparent age from a single image," ICCV, 2015.

[2] R. Rothe, R. Timofte, and L. V. Gool, "Deep expectation of real and apparent age from a single image
without facial landmarks," IJCV, 2016.
