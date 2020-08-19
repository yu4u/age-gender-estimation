from pathlib import Path
import numpy as np
import cv2
import albumentations as A
from tensorflow.keras.utils import Sequence


transforms = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.03125, scale_limit=0.20, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT,
                       value=0, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5)
])


class ImageSequence(Sequence):
    def __init__(self, cfg, df, mode):
        self.df = df
        self.indices = np.arange(len(df))
        self.batch_size = cfg.train.batch_size
        self.img_dir = Path(__file__).resolve().parents[1].joinpath("data", f"{cfg.data.db}_crop")
        self.img_size = cfg.model.img_size
        self.mode = mode

    def __getitem__(self, idx):
        sample_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs = []
        genders = []
        ages = []

        for _, row in self.df.iloc[sample_indices].iterrows():
            img = cv2.imread(str(self.img_dir.joinpath(row["img_paths"])))
            img = cv2.resize(img, (self.img_size, self.img_size))

            if self.mode == "train":
                img = transforms(image=img)["image"]

            imgs.append(img)
            genders.append(row["genders"])
            ages.append(row["ages"])

        imgs = np.asarray(imgs)
        genders = np.asarray(genders)
        ages = np.asarray(ages)

        return imgs, (genders, ages)

    def __len__(self):
        return len(self.df) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
