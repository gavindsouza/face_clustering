"""
Make shift testing file
probs use unittest here
"""
from core.face_clustering import Model
from core.face_encoding import encode_all, save_encodes
from db.SQLite3 import SQLite

import pickle

if __name__ == "__main__":
    print("[IMP] This is only completed to the demo stage. Make sure test folder has no sub dirs or other files, only images. Results saved in 'temp_files/results.csv'")
    IMG_FOLDER = "/home/gavin/Pictures" or input("Input Absolute path of folder: ")   # /mnt/FOURTH/data/kaggle/faces-data/

    db = SQLite()
    print("DB instance created")

    encodes = encode_all(IMG_FOLDER, verbose=True)
    # if already encoded and saved to pickle:
    # encodes = pickle.loads(open('test-encodes.pkl', 'rb').read())
    print("All images encoded")

    db.enter_batch_encodings(encodes)
    print("Saved all encodes to DB")

    encodes = db.get_encodes()
    print("Received all from DB")

    model = Model("dbscan")
    model.load_data(from_db=encodes)
    print("Fit data on model")

    predicted = model.predict()
    print("Predicted lebels")

    model.save_csv()
    print("Saved CSV")

    """
    # Plot result
    import matplotlib.pyplot as plt
    import numpy as np

    # Black removed and is used for noise instead.
    unique_labels = set(model.clt.labels_)

    # Fix X and core_samples_mask
    X, core_sameples_mask = to_be_done()

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (model.clt.labels_ == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    """