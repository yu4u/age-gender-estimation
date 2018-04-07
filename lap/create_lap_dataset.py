import argparse
import better_exceptions
from pathlib import Path
import zipfile
import urllib.request


zip_names = ["train_1.zip", "train_2.zip", "train_gt.zip", "valid.zip", "valid_gt.zip"]
urls = ["http://***/train_1.zip",
        "http://***/train_2.zip",
        "http://***/train_gt.zip",
        "http://***/valid.zip",
        "http://***/valid_gt.zip"]
gt_pwd = b"***"

dataset_root = Path(__file__).resolve().parent.joinpath("dataset")


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


def download():
    for zip_name, url in zip(zip_names, urls):
        local_path = dataset_root.joinpath(zip_name)
        urllib.request.urlretrieve(url, str(local_path))


def crop():
    pass


def extract():
    for zip_name in zip_names:
        zip_path = dataset_root.joinpath(zip_name)
        password = None

        if not zip_path.is_file():
            raise RuntimeError("{} was not found. Please download the LAP dataset.".format(zip_name))

        with zipfile.ZipFile(str(zip_path), "r") as f:

            if zip_name in ["train_1.zip", "train_2.zip"]:
                extract_path = dataset_root.joinpath("train_images")
            elif zip_name == "valid.zip":
                extract_path = dataset_root.joinpath("validation_images")
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
