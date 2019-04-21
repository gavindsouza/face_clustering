# imports - standard imports
import pickle
import os
import sys
from multiprocessing import cpu_count as num_jobs

# imports - third party imports
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from imutils import build_montages
import numpy as np


class Model:
	def __init__(self, algorithm: str = "dbscan"):
		choices = {
			"dbscan" : DBSCAN(metric="euclidean", n_jobs=num_jobs()),
			"kmeans" : KMeans(precompute_distances='auto', n_jobs=num_jobs()),
			"meanshift" : MeanShift(n_jobs=num_jobs())
		}
		self.clt = choices[algorithm]


	def load_data(self, data: list = None, pickle_path: str = None):
		if data is not None:
			try:
				self.data = data
				self.encodings = [row["encoding"] for row in self.data]
			except:
				print("Unexpected error:", sys.exc_info()[0])
				raise
				
		elif pickle_path is not None:
			file = open(pickle_path, "rb").read()
			self.data = pickle.loads(file)
			self.encodings = [row["encoding"] for row in self.data]
			print("[INFO] Encodings Loaded!")
		
		else:
			print("Need an input of encoded images")


	def predict(self, enc: list=None):
		if enc is None:
			enc = self.encodings

		print("[INFO] clustering...")
		
		self.clt.fit(enc)
		self.predicted_labels = [(label, whole_data) for label, whole_data in zip(self.clt.labels_, self.data)]

		return self.predicted_labels

	
	def save_csv(self, labels: list = None, filename: str = "results"):
		if labels is None:
			labels = self.predicted_labels

		with open(f"temp_files/{filename}.csv", "w") as f:
			for label_ in labels:
				label, whole_data = label_

				img_path = whole_data["image_path"]
				time_stamp = whole_data["time_stamp"]
				box_loc = whole_data["box_loc"]
				encoding = whole_data["encoding"]
				numeric_label = label

				f.write(f"{img_path},{box_loc},{time_stamp},{numeric_label}\n")
