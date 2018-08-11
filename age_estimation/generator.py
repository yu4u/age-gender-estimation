import better_exceptions
import random
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from keras.utils import Sequence, to_categorical
import Augmentor


def get_transform_func():
    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.95)
    p.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.5, rectangle_area=0.2)

    def transform_image(image):
        image = [Image.fromarray(image)]
        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return image[0]
    return transform_image


class FaceGenerator(Sequence):
    def __init__(self, appa_dir, utk_dir=None, batch_size=32, image_size=224):
        self.image_path_and_age = []
        self._load_appa(appa_dir)

        if utk_dir:
            self._load_utk(utk_dir)

        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.random.permutation(self.image_num)
        self.transform_image = get_transform_func()

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)

        sample_indices = self.indices[idx * batch_size:(idx + 1) * batch_size]

        for i, sample_id in enumerate(sample_indices):
            image_path, age = self.image_path_and_age[sample_id]
            image = cv2.imread(str(image_path))
            x[i] = self.transform_image(cv2.resize(image, (image_size, image_size)))
            y[i] = age

        return x, to_categorical(y, 101)

    def on_epoch_end(self):
        self.indices = np.random.permutation(self.image_num)

    def _load_appa(self, appa_dir):
        appa_root = Path(appa_dir)
        train_image_dir = appa_root.joinpath("train")
        gt_train_path = appa_root.joinpath("gt_avg_train.csv")
        df = pd.read_csv(str(gt_train_path))

        for i, row in df.iterrows():
            age = min(100, int(row.apparent_age_avg))
            # age = int(row.real_age)
            image_path = train_image_dir.joinpath(row.file_name + "_face.jpg")

            if image_path.is_file():
                self.image_path_and_age.append([str(image_path), age])

    def _load_utk(self, utk_dir):
        image_dir = Path(utk_dir)

        for image_path in image_dir.glob("*.jpg"):
            image_name = image_path.name  # [age]_[gender]_[race]_[date&time].jpg
            age = min(100, int(image_name.split("_")[0]))

            if image_path.is_file():
                self.image_path_and_age.append([str(image_path), age])


class ValGenerator(Sequence):
    def __init__(self, appa_dir, batch_size=32, image_size=224):
        self.image_path_and_age = []
        self._load_appa(appa_dir)
        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, 1), dtype=np.int32)

        for i in range(batch_size):
            image_path, age = self.image_path_and_age[idx * batch_size + i]
            image = cv2.imread(str(image_path))
            x[i] = cv2.resize(image, (image_size, image_size))
            y[i] = age

        return x, to_categorical(y, 101)

    def _load_appa(self, appa_dir):
        appa_root = Path(appa_dir)
        val_image_dir = appa_root.joinpath("valid")
        gt_val_path = appa_root.joinpath("gt_avg_valid.csv")
        df = pd.read_csv(str(gt_val_path))

        for i, row in df.iterrows():
            age = min(100, int(row.apparent_age_avg))
            # age = int(row.real_age)
            image_path = val_image_dir.joinpath(row.file_name + "_face.jpg")

            if image_path.is_file():
                self.image_path_and_age.append([str(image_path), age])
