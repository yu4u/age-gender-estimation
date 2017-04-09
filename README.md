# Age and Gender Estimation
This is a Keras implementation of a CNN network for estimating age and gender from a face image.
In training, [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) is used.

## Dependencies
- Python3.5+
- Keras
- scipy, numpy, Pandas, tqdm
- OpenCV3

## Usage

Download the dataset. The dataset is downloaded and extracted to the `data` directory.

```sh
./download.sh
```

Filter out noise data and serialize images and labels for training into `.mat` file.
Please check `check_dataset.ipynb` for the details of the dataset.
```sh
python create_db.py --output data/imdb_db.mat --db imdb --img_size 64
```

Train the network using the training data created above.

```sh
python3 train.py --input data/imdb_db.mat
```

## Network architecture
In [the original paper](https://www.vision.ee.ethz.ch/en/publications/papers/articles/eth_biwi_01299.pdf), the pretrained VGG network is adopted.
Here the Wide Residual Network (WideResNet) is trained from scratch.
I modified the @asmith26's implementation of the WideResNet; two classification layers (for age and gender estimation) are added on the top of the WideResNet.
Note that age and gender are estimated independently using different two CNNs.

