from core.face_clustering import Model
from core.face_encoding import encode_all, save_encodes

if __name__ == "__main__":
    encodes = encode_all("/mnt/FOURTH/data/kaggle/faces-data/", verbose=True)
    # save_encodes(encodes=encodes, verbose=True)
    model = Model("kmeans")
    # model.load_data(db_path="/mnt/SECOND/Code/projects/face_clustering/face_clustering/temp_files/faces.pickle")
    model.load_data(data=encodes)
    model.predict()
    model.save_data()