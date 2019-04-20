from core.face_clustering import Model

model = Model("kmeans")
model.load_data(db_path="/mnt/SECOND/Code/projects/face_clustering/faces.pickle")
model.predict()
model.save_data()