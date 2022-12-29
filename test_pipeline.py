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
    logging.basicConfig(level=logging.INFO)

    logger.info("[IMP] This is only completed to the demo stage. Make sure test folder has no sub dirs or other files, only images. Results saved in 'temp_files/results.csv'")

    IMG_FOLDER = "/home/gavin/Pictures" or input("Input Absolute path of folder: ")   # /mnt/FOURTH/data/kaggle/faces-data/

    db = SQLite()
    logger.info("DB instance created")

    encodes = encode_all(IMG_FOLDER, verbose=True)
    logger.info("All images encoded")

    db.enter_batch_encodings(encodes)
    logger.info("Saved all encodes to DB")

    encodes = db.get_encodes()
    logger.info("Received all from DB")

    model = Model("dbscan")
    model.load_data(from_db=encodes)
    logger.info("Fit data on model")

    predicted = model.predict()
    logger.info("Predicted lebels")

    model.save_csv()
    logger.info("Saved CSV")

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