from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

from src.utils import get_meta


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default="imdb",
                        help="dataset; wiki or imdb")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    db = args.db
    min_score = args.min_score
    root_dir = Path(__file__).parent
    data_dir = root_dir.joinpath("data", f"{db}_crop")
    mat_path = data_dir.joinpath(f"{db}.mat")
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

    genders = []
    ages = []
    img_paths = []
    sample_num = len(face_score)

    for i in tqdm(range(sample_num)):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        genders.append(int(gender[i]))
        ages.append(age[i])
        img_paths.append(full_path[i][0])

    outputs = dict(genders=genders, ages=ages, img_paths=img_paths)
    output_dir = root_dir.joinpath("meta")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir.joinpath(f"{db}.csv")
    df = pd.DataFrame(data=outputs)
    df.to_csv(str(output_path), index=False)


if __name__ == '__main__':
    main()
