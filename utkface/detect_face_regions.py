import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import cv2
import dlib


def get_args():
    parser = argparse.ArgumentParser(description="This script detect faces using dlib and save detected face rects",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="path to input directory that includes part1, part2, part3 sub-directories")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output csv file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    root_dir = Path(args.input)
    output_path = args.output
    detector = dlib.get_frontal_face_detector()
    results = []

    for image_path in tqdm(root_dir.glob("*/*.jpg")):
        img = cv2.imread(str(image_path))
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detected = detector(input_img, 1)

        if len(detected) != 1:
            continue

        d = detected[0]
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        image_name = image_path.name
        results.append([image_name, x1, y1, x2, y2])

    pd.DataFrame(data=results, columns=["image_name", "x1", "y1", "x2", "y2"]).to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
