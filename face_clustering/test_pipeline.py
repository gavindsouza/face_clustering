"""
Make shift testing file
probs use unittest here
"""
from core.face_clustering import Model
from core.face_encoding import encode_all, save_encodes
import pickle
from pprint import pprint as print


if __name__ == "__main__":
    # encodes = encode_all("/mnt/FOURTH/data/kaggle/faces-data/", verbose=True)
    # save_encodes(encodes=encodes, verbose=True)
    model = Model("meanshift")
    model.load_data(pickle_path="/mnt/SECOND/Code/projects/face_clustering/face_clustering/temp_files/faces.pickle")
    # model.load_data(data=encodes)
    predicted = model.predict()
    test_data = model.predicted_labels[23]
    print(test_data[0])
    print(test_data[1])
    print(test_data[1]['encoding'])
    print(test_data[1]['time_stamp'])
    print(predicted[0])
    model.save_csv()
