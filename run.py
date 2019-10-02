from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import time
import pandas as pd
import boto3
import uuid
import face_recognition
from settings import *

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    global cap
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))

def upload(content):
	bucket = 'adil-test'
	filename = 'age-gender-estimation/{0}.txt'.format(uuid.uuid1())
	s3.Bucket(bucket).put_object(Key= filename, Body=content, ACL='public-read')

def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings

def main():
    global s3
    # s3 = boto3.client('s3')
    session = boto3.Session(
	    aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
	    aws_secret_access_key=AWS_SERVER_SECRET_KEY)
    s3 = session.resource('s3')    
    
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file
    margin = args.margin
    image_dir = args.image_dir

    if not weight_file:
        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()
    print('OK. Ready!')


    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]-5),
                                        "M" if predicted_genders[i][0] < 0.5 else "F")
                draw_label(img, (d.left(), d.top()), label)


        cv2.imshow("result", img)
        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        if key == 27:  # ESC
            break

        if key == 32:  # SPACE
            frame_id = int(time.time())

            # _,embeddings = get_face_embeddings_from_image(img)
            # embedded = len(embeddings)==len(detected)
            # if not embedded:
            #     embeddings =  [None]*len(detected)
            #     print('Oops. Please try again!')
            #     return None
            

            response=[[
                        str(frame_id)+str(i+1).zfill(2),
                        str(pd.to_datetime('now')),
                        str(frame_id),
                        str(i+1),
                        str(int(predicted_ages[i])-5),
                        "M" if predicted_genders[i][0] < 0.5 else "F"
                        # ,str(embeddings[i])
                    ] for i, d in enumerate(detected)]            

            j_response=[{

                        'key':int(str(frame_id)+str(i+1).zfill(2)),
                        'datetime':str(pd.to_datetime('now')),
                        'frame_id':frame_id,
                        'frame_face_id':i+1,
                        'predicted_ages':str(int(predicted_ages[i])-5),
                        'predicted_genders':"M" if predicted_genders[i][0] < 0.5 else "F"
                        # ,'embeddings':list(embeddings[i])
                        } for i, d in enumerate(detected)]   

            output = '\n'.join(['|'.join(i) for i in response])
            j_output = '\n'.join([str(j) for j in j_response])
            print(j_output)
            try:
            	upload(j_output)
            except:
            	print('Connection to S3 failed.')

        if key == 115:  # s
        	print('Please enter email address to send result to (Consent required).')
        	print('Sending to '+ input(),end='...\n')
	       	output = '\n'.join(['\t'.join(i) for i in response])
        	print(output)
        	# upload(output)
def remain():
    while True:
        try:
            main()
        except: 
            cap.release()
            cv2.destroyAllWindows()
            continue

if __name__ == '__main__':
    remain()
