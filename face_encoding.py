# imports - standard imports
import time
import datetime
import os
import pickle
import multiprocessing
from multiprocessing import Manager

# imports - third party imports
import face_recognition
from PIL import Image
import numpy as np


DATA_PATH = r"/mnt/FOURTH/data/kaggle/faces-data/"
# PICKLE_PATH = os.path.join(DATA_PATH, "faces.pickle")
PICKLE_PATH = "faces.pickle"
IMG_LIST = os.listdir(DATA_PATH)
TOTAL_IMG = len(IMG_LIST)
num_process = multiprocessing.cpu_count()
manager = Manager()
ALL_DATA = manager.list()


def make_data(img_name):
    img_path = os.path.join(DATA_PATH, img_name)
    img = np.asarray(Image.open(img_path))

    boxes = face_recognition.face_locations(img, model='cnn')
    encodings = face_recognition.face_encodings(img, boxes)

    data_img = [
        {
            "time_stamp": datetime.datetime.utcnow(),
            "image_path": img_path,
            "box_loc": box,
            "encoding": encode
        }
        for (box, encode) in zip(boxes, encodings)]

    ALL_DATA.extend(data_img)

    PROGRESS = len(ALL_DATA)
    if PROGRESS % 100 == 0:
        print(f"{PROGRESS}/{TOTAL_IMG} images done...")


if __name__ == "__main__":

    start = time.time()
    with multiprocessing.Pool(num_process) as pool:
        pool.map(make_data, IMG_LIST)
    end = time.time()
    print(f"{len(ALL_DATA)} images in {end - start}s")

    start = time.time()
    new_list = [x for x in ALL_DATA]
    end = time.time()
    print(f"List converted in {end - start}s")

    with open(PICKLE_PATH, 'wb') as f:
        f.write(pickle.dumps(new_list))
