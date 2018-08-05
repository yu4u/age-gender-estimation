import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import dlib


def get_args():
    parser = argparse.ArgumentParser(description="This script detect faces using dlib and save detected faces",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input directory that includes part1, part2, part3 sub-directories")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output directory")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around face regions")
    args = parser.parse_args()
    return args


# robust image cropping from
# https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
def imcrop(img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


def main():
    args = get_args()
    root_dir = Path(args.input)
    output_dir = Path(args.output)
    margin = args.margin
    detector = dlib.get_frontal_face_detector()
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(root_dir.glob("*/*.jpg")):
        img = cv2.imread(str(image_path))
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detected = detector(input_img, 1)

        if len(detected) != 1:
            continue

        d = detected[0]
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        xw1 = int(x1 - margin * w)
        yw1 = int(y1 - margin * h)
        xw2 = int(x2 + margin * w)
        yw2 = int(y2 + margin * h)
        image_name = image_path.name
        cropped_img = imcrop(img, xw1, yw1, xw2, yw2)
        cv2.imwrite(str(output_dir.joinpath(image_name)), cropped_img)


if __name__ == '__main__':
    main()
