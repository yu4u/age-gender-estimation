# APPA-REAL Dataset

Use [APPA-REAL Dataset](http://chalearnlap.cvc.uab.es/dataset/26/description/) for finetuning CNNs for age estimation.

> The APPA-REAL database contains 7,591 images with associated real and apparent age labels. The total number of apparent votes is around 250,000. On average we have around 38 votes per each image and this makes the average apparent age very stable (0.3 standard error of the mean).

## Download Dataset

1. Move to `age-gender-estimation/appa-real` directory.
2. Download and extract the  APPA-REAL dataset:

```bash
wget http://158.109.8.102/AppaRealAge/appa-real-release.zip
unzip appa-real-release.zip
```

After that, the directory tree becomes like this:

```
age-gender-estimation
├── appa-real
│   ├── appa-real-release
│   │   ├── test
│   │   ├── train
│   │   └── valid
│   └── ignored_images
...
```

## Ignored List
The dataset includes cropped and rotated face images with a 40% margin obtained from a face detector.

I manually checked the dataset, and found several problems:
1. It incldues non-face images.
2. There are many inappropriate cropped images due to failure in face detection.
3. Multiple identities (faces) in a single image  with no priority.

Therefore, I created [ignored list](ignore_list.txt) to exclude these inappropriate images (only for training set).

The examples of these images are:

<img src="ignored_images/002460.jpg" width="320px">
<img src="ignored_images/002630.jpg" width="320px">
<img src="ignored_images/002633.jpg" width="320px">

Please refer to [check_ignore_list.ipynb](check_ignore_list.ipynb) for more examples.
