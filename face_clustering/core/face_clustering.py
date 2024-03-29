# imports - standard imports
import csv
import os
import pickle
import sys
import logging
from multiprocessing import cpu_count as num_jobs

# imports - library imports
from .. import __name__ as NAME

# imports - third party imports
import numpy as np
from imutils import build_montages
from sklearn.cluster import DBSCAN, KMeans, MeanShift

logger = logging.getLogger(NAME)

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
			logger.info("Definitely Untested")
			self.data, self.encodings = list(zip(*from_db))

		elif data is not None:
			try:
				self.data = data
				self.encodings = [row["encoding"] for row in self.data]
			except Exception:
				logger.info("Unexpected error:", sys.exc_info()[0])
				raise

		elif pickle_path is not None:
			file = open(pickle_path, "rb").read()
			self.data = pickle.loads(file)
			self.encodings = [row["encoding"] for row in self.data]
			logger.info("Encodings Loaded!")

		else:
			logger.info("Need an input of encoded images")

	def predict(self, enc: list=None):
		if enc is None:
			enc = self.encodings

		logger.info("clustering input data...")

		self.clt.fit(enc)
		self.predicted_labels = list(zip(self.clt.labels_, self.data))

		logger.info(f"Predicted {len(self.predicted_labels)} encodings...")

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
		if dbms == 'postgres':
			logger.info("Well this isn't implemented yet\nWill be if proved advantageous to do so")
			raise NotImplementedError

		elif dbms == 'sqlite':
			from face_clustering.db.SQLite3 import SQLite
			db = SQLite()

			for row in self.predicted_labels:
				label, whole_data = row

				numeric_label = label

				img_path = whole_data["image_path"]
				time_stamp = whole_data["time_stamp"]
				box_loc = whole_data["box_loc"]
				encoding = whole_data["encoding"]
				db.entry(
					img_path=img_path,
					location_of_face=box_loc,
					encoding=encoding,
					time_stamp=time_stamp
				)

		else:
			logger.info("Select between 'sqlite' and 'postgres'")
			raise NotImplementedError
