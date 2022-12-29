"""
Make shift testing file
probs use unittest here
"""
import logging

from face_clustering.core.face_clustering import Model
from face_clustering.core.face_encoding import encode_all
from face_clustering.db.SQLite3 import SQLite


if __name__ == "__main__":
    logger = logging.getLogger("test_pipeline")
    logging.basicConfig(level=logging.NOTSET)

    logger.info("This is only completed to the demo stage. Results saved in 'temp_files/results.csv'")

    IMG_FOLDER = "/home/gavin/Pictures" or input("Input Absolute path of folder: ")   # /mnt/FOURTH/data/kaggle/faces-data/

    with SQLite() as db:
        encodes = encode_all(IMG_FOLDER, verbose=True)
        db.enter_batch_encodings(encodes)
        encodes = db.get_encodes()

        model = Model("dbscan")
        model.load_data(from_db=encodes)

        predictions = model.predict()
        model.save_csv()

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