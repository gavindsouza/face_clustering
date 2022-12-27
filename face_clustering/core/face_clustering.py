# imports - standard imports
import csv
import os
import pickle
import sys
from multiprocessing import cpu_count as num_jobs

# imports - third party imports
import numpy as np
from imutils import build_montages
from sklearn.cluster import DBSCAN, KMeans, MeanShift


class Model:
	def __init__(self, algorithm: str = "dbscan"):
		choices = {
			"dbscan" : DBSCAN(metric="euclidean", n_jobs=num_jobs()),
			"kmeans" : KMeans(),
			"meanshift" : MeanShift(n_jobs=num_jobs())
		}
		self.clt = choices[algorithm]

	def load_data(self, from_db: list = None, data: list = None, pickle_path: str = None):
		"""
		::params:: type = default
		from_db: tuple = None
		data: list = None
		pickle_path: str = None
		"""
		if from_db is not None:
			print("Definitely Untested")
			self.data, self.encodings = list(zip(*from_db))

		elif data is not None:
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
		self.predicted_labels = list(zip(self.clt.labels_, self.data))

		return self.predicted_labels

	def save_csv(self, labels: list = None, filename: str = "results"):
		if labels is None:
			labels = self.predicted_labels

		os.makedirs("temp_files", exist_ok=True)
		with open(f"temp_files/{filename}.csv", "w") as f:
			csv_out = csv.writer(f)
			csv_out.writerow(['label', 'data'])

			for data in self.predicted_labels:
				csv_out.writerow(data)

	def save_db(self, dbms: str = 'sqlite'):
		print("Changes to be made\nUnfinished!")
		return
