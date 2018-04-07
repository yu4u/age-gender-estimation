import argparse
import better_exceptions
import sys
import time
from pathlib import Path
import zipfile
import bz2
import urllib.request
import dlib
import cv2


zip_names = ["train_1.zip", "train_2.zip", "train_gt.zip", "valid.zip", "valid_gt.zip"]
urls = ["http://***/train_1.zip",
        "http://***/train_2.zip",
        "http://***/train_gt.zip",
        "http://***/valid.zip",
        "http://***/valid_gt.zip"]
gt_pwd = b"***"

dataset_root = Path(__file__).resolve().parent.joinpath("dataset")
model_root = Path(__file__).resolve().parent.joinpath("model")
train_image_dir = dataset_root.joinpath("train_images")
validation_image_dir = dataset_root.joinpath("validation_images")
train_crop_dir = dataset_root.joinpath("train_crop")
validation_crop_dir = dataset_root.joinpath("validation_crop")


def get_args():
    parser = argparse.ArgumentParser(description="This script downloads the LAP dataset "
                                                 "and preprocess for training and evaluation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(help="subcommands", dest="subcommand")
    subparsers.add_parser("download", help="Downdload the LAP dataset")
    subparsers.add_parser("extract", help="Unzip the LAP dataset")
    subparsers.add_parser("crop", help="Crop face regions using dlib")
    args = parser.parse_args()

    return parser, args


def reporthook(count, block_size, total_size):
    global start_time

    if count == 0:
        start_time = time.time()
        return

    duration = int(time.time() - start_time)
    current_size = count * block_size
    remaining_size = total_size - current_size
    speed = int(current_size / (1024 * duration + 1))
    percent = min(int(count * block_size * 100 / total_size), 100)
    remaining_time = int(duration * (remaining_size / current_size))
    sys.stdout.write("\r{}%, {:6.2f}/{:6.2f}MB, {}KB/s, passed: {}s, remaining: {}s".format(
        percent, current_size / (1024 * 1024), total_size / (1024 * 1024), speed, duration, remaining_time))
    sys.stdout.flush()


def download():
    dataset_root.mkdir(parents=True, exist_ok=True)  # requires Python 3.5 or above

    for zip_name, url in zip(zip_names, urls):
        print("downloading {}".format(zip_name))
        local_path = dataset_root.joinpath(zip_name)
        urllib.request.urlretrieve(url, str(local_path), reporthook)


def crop():
    detector_model_path = model_root.joinpath("mmod_human_face_detector.dat")

    if not detector_model_path.is_file():
        model_root.mkdir(parents=True, exist_ok=True)  # requires Python 3.5 or above
        detector_model_url = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
        detector_model_bz2 = str(detector_model_path) + ".bz2"
        print("downloading {}".format(detector_model_path.name))
        urllib.request.urlretrieve(detector_model_url, detector_model_bz2, reporthook)

        with open(detector_model_bz2, "rb") as source, open(str(detector_model_path), "wb") as dest:
            dest.write(bz2.decompress(source.read()))

    detector = dlib.cnn_face_detection_model_v1(str(detector_model_path))

    for image_dir, crop_dir in [[train_image_dir, train_crop_dir], [validation_image_dir, validation_crop_dir]]:
        for image_path in image_dir.glob("*.jpg"):
            frame = cv2.imread(str(image_path))
            img_h, img_w, _ = frame.shape
            factor = 800 / max(img_h, img_w)
            frame_resized = cv2.resize(frame, None, fx=factor, fy=factor)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            dets = detector(frame_rgb, 1)

            if len(dets) != 1:
                print("{} faces were detected for {}".format(len(dets), image_path.name))
                rects = [[d.rect.left(), d.rect.right(), d.rect.top(), d.rect.bottom()] for d in dets]
                print(rects)


def extract():
    for zip_name in zip_names:
        zip_path = dataset_root.joinpath(zip_name)
        password = None

        if not zip_path.is_file():
            raise RuntimeError("{} was not found. Please download the LAP dataset.".format(zip_name))

        with zipfile.ZipFile(str(zip_path), "r") as f:

            if zip_name in ["train_1.zip", "train_2.zip"]:
                extract_path = train_image_dir
            elif zip_name == "valid.zip":
                extract_path = validation_image_dir
            else:
                extract_path = dataset_root

            if zip_name == "valid_gt.zip":
                password = gt_pwd

            extract_path.mkdir(parents=True, exist_ok=True)  # requires Python 3.5 or above
            f.extractall(path=str(extract_path), pwd=password)


def main():
    parser, args = get_args()

    if args.subcommand == "download":
        download()
    elif args.subcommand == "extract":
        extract()
    elif args.subcommand == "crop":
        crop()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
