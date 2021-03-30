# imports - standard imports
import datetime
import itertools
import multiprocessing
import os
import pickle
import sys
import time

# imports - third party imports
import face_recognition
import numpy as np
from PIL import Image

PROJECT_PATH = os.path.abspath(os.path.split(sys.argv[0])[0])
num_process = multiprocessing.cpu_count()
manager = multiprocessing.Manager()
ALL_DATA = manager.list()


def encode_one(img_path: str, total_img: int = None, verbose: bool = False):
    img = np.asarray(Image.open(img_path))

    boxes = face_recognition.face_locations(img, model='hog')
    encodings = face_recognition.face_encodings(img, boxes)

    # data_img = [
    #     {
    #         "encoded_time_stamp": str(datetime.datetime.utcnow()),
    #         "image_path": img_path,
    #         "box_loc": box,
    #         "encoding": encode
    #     }
    #     for (box, encode) in zip(boxes, encodings)]

    data_img = [
        (
            img_path,
            encoding
        ) for encoding in encodings
    ]

    ALL_DATA.extend(data_img)

    if verbose:
        progress = len(ALL_DATA)
        if progress % 100 == 0:
            print(f"{progress} faces from {total_img} images done...")


def encode_all(data_path: str, verbose: bool = False):
    img_dir = os.listdir(data_path)
    img_list = (os.path.join(data_path, img_name) for img_name in img_dir)
    total_img = len(img_dir)

    start = time.time()
    with multiprocessing.Pool(num_process) as pool:
        pool.starmap(encode_one, zip(img_list, itertools.repeat(
            total_img, total_img), itertools.repeat(verbose, total_img)))
    end = time.time()

    if verbose:
        print(f"{len(ALL_DATA)} images in {end - start}s")

    start = time.time()
    end = time.time()

    if verbose:
        print(f"List converted in {end - start}s")

    return [x for x in ALL_DATA]


def save_encodes(encodes: list, pickle_path: str = None, verbose: bool = False):
    if pickle_path is None:
        pickle_path = os.path.join(PROJECT_PATH, "temp_files", "faces.pickle")

    with open(pickle_path, 'wb') as f:
        f.write(pickle.dumps(encodes))

    if verbose:
        print(f"Results of encoding saved as {pickle_path}")
