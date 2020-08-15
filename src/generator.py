from pathlib import Path
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence, to_categorical


class ImageSequence(Sequence):
    def __init__(self, cfg, df):
        self.df = df
        self.indices = np.arange(len(df))
        self.batch_size = cfg.model.batch_size
        self.img_dir = Path(__file__).resolve().parents[1].joinpath("data", f"{cfg.data.db}_crop")
        self.img_size = cfg.model.img_size

    def __getitem__(self, idx):
        sample_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs = []
        genders = []
        ages = []

        for _, row in self.df.iloc[sample_indices].iterrows():
            img = cv2.imread(str(self.img_dir.joinpath(row["img_paths"])))
            img = cv2.resize(img, (self.img_size, self.img_size))
            imgs.append(img)
            genders.append(row["genders"])
            ages.append(row["ages"])

        imgs = np.asarray(imgs)
        genders = to_categorical(genders, 2)
        ages = to_categorical(ages, 101)

        return imgs, (genders, ages)

    def __len__(self):
        return len(self.df) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
