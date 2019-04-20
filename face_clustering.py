# imports - standard imports
import pickle
import os
from multiprocessing import cpu_count as num_jobs

# imports - third party imports
import cv2
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np


data_path = r"/mnt/FOURTH/data/kaggle/faces-data/"
faces = []


print("[INFO] loading encodings...")
data = pickle.loads(open("faces_hog.pickle", "rb").read())
encodings = [row["encoding"] for row in data]

# cluster the embeddings
print("[INFO] clustering...")
clt = DBSCAN(metric="euclidean", n_jobs=num_jobs())
clt.fit(encodings)

labels = [(label, whole_data) for label, whole_data in zip(clt.labels_, data)]

with open("final.csv", "w") as f:
	for label in labels:
		label, whole_data = label
		
		img_path = whole_data["img_path"],
		time_stamp = whole_data["time_stamp"],
		box_loc = whole_data["box_loc"],
		encoding = whole_data["encoding"],
		numeric_label = label
        
		f.write(f"{img_path},{box_loc},{time_stamp},{numeric_label}\n")
