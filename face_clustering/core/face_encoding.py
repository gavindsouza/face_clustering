# imports - standard imports
import multiprocessing
import os
import pickle
import time
from itertools import repeat
from pathlib import Path
import logging

# imports - library imports
from .. import __name__ as NAME

# imports - third party imports
import face_recognition
import numpy as np
from PIL import Image

CWD = Path(os.getcwd())
num_process = multiprocessing.cpu_count() // 2
manager = multiprocessing.Manager()
ALL_DATA = manager.list()
logger = logging.getLogger(NAME)


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
            logger.info(f"{progress} faces from {total_img} images done...")


def encode_all(data_path: str, verbose: bool = False):
    img_dir = os.listdir(data_path)
    img_list = [os.path.join(data_path, img_name) for img_name in img_dir if img_name.lower().endswith((".jpg", ".png", ".jpeg"))]
    total_img = len(img_list)
    args = list(zip(img_list, repeat(total_img, total_img), repeat(verbose, total_img)))

    start = time.time()
    with multiprocessing.Pool(num_process) as pool:
        pool.starmap(encode_one, iterable=args)
    end = time.time()

    if verbose:
        logger.info(f"{len(ALL_DATA)} images in {end - start}s")

    return [x for x in ALL_DATA]


def save_encodes(encodes: list, pickle_path: str = None, verbose: bool = False):
    if pickle_path is None:
        pickle_path = CWD / "temp_files" / "faces.pickle"

    with open(pickle_path, 'wb') as f:
        f.write(pickle.dumps(encodes))

    if verbose:
        logger.info(f"Results of encoding saved as {pickle_path}")
