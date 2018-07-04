import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import scipy.io
import cv2


def get_args():
    parser = argparse.ArgumentParser(description="This script creates database for training from the UTKFace dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to the UTKFace image directory")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output database mat file")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    image_dir = Path(args.input)
    output_path = args.output
    img_size = args.img_size

    out_genders = []
    out_ages = []
    out_imgs = []

    for i, image_path in enumerate(tqdm(image_dir.glob("*.jpg"))):
        image_name = image_path.name  # [age]_[gender]_[race]_[date&time].jpg
        age, gender = image_name.split("_")[:2]
        out_genders.append(int(gender))
        out_ages.append(min(int(age), 100))
        img = cv2.imread(str(image_path))
        out_imgs.append(cv2.resize(img, (img_size, img_size)))

    output = {"image": np.array(out_imgs), "gender": np.array(out_genders), "age": np.array(out_ages),
              "db": "utk", "img_size": img_size, "min_score": -1}
    scipy.io.savemat(output_path, output)


if __name__ == '__main__':
    main()
