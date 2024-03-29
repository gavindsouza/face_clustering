# imports - standard imports
import pickle
import sqlite3
import logging

# imports - library imports
from .. import __name__ as NAME

# imports - third party imports
import numpy as np

logger = logging.getLogger(NAME)

class SQLite:
    """
    The commented queries are for the extension
    Current implementation: Bare minimum
    """
    def __init__(self, db_name = ":memory:"):
        self.db_name = db_name

        if self.db_name == ":memory:":
            logger.debug("DB exists on primary memory, Encodings will not be saved !!!")

        self.connection = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()

        sqlite3.register_adapter(np.array, SQLite.adapt_array)
        sqlite3.register_converter("array", SQLite.convert_array)

        sqlite3.register_adapter(tuple, SQLite.adapt_tuple)    #cannot use pickle.dumps directly because of inadequate argument signature
        sqlite3.register_converter("tuple", pickle.loads)

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS encodings (
                img_path TEXT,
                encoding array
            )"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS labels (
                img_path TEXT PRIMARY KEY,
                label NUMBER
            )"""
        )

    def __post_init__(self):
        logger.info(f"DB instance created as {self}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cursor.close()
        self.connection.close()
        logger.info("DB connection closed")

    def enter_label(self, img_path: str, label: int, location_of_face: tuple = None, encoding: np.array = None, time_stamp: object = None):
        # query = "INSERT INTO labels (img_path, loc_box, encoding, time_stamp) VALUES (?, ?, ?, ?)"
        # self.cursor.execute(query, (img_path, location_of_face, encoding, time_stamp))
        try:
            self.cursor.execute("INSERT INTO labels (img_path, encoding) VALUES (?, ?)", (img_path, label))
            self.connection.commit()

        except sqlite3.IntegrityError as integrityMessage:
            logger.exception(integrityMessage)

    def enter_encoding(self, img_path: str, encoding: np.array, location_of_face: tuple = None, time_stamp: object = None):
        # query = "INSERT INTO encodings (img_path, loc_box, encoding, time_stamp) VALUES (?, ?, ?, ?)"
        # self.cursor.execute(query, (img_path, location_of_face, encoding, time_stamp))
        try:
            self.cursor.execute("INSERT INTO encodings (img_path, encoding) VALUES (?, ?)", (img_path, encoding))
            self.connection.commit()

        except sqlite3.IntegrityError as integrityMessage:
            logger.exception(integrityMessage)

    def enter_batch_encodings(self, data: list, location_of_face: tuple = None, time_stamp: object = None):
        """
        data: list(*tuple)
            img_path: str
            encoding: np.array

        data = [("path_to_img", np.array), ... ("path_to_img", np.array)]
        """
        try:
            self.cursor.executemany("INSERT INTO encodings (img_path, encoding) VALUES (?, ?)", (data))
            self.connection.commit()
            logger.info(f"Saved {len(data)} encodings to DB")

        except sqlite3.IntegrityError as integrityMessage:
            logger.exception(integrityMessage)

    def get_encodes(self):
        self.cursor.execute("SELECT * FROM encodings")
        encodings = self.cursor.fetchall()
        logger.info(f"Fetched {len(encodings)} encodings from DB")
        return encodings

    @staticmethod
    def adapt_array(arr):
        # for other possible methods: http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        return arr.tobytes()

    @staticmethod
    def convert_array(text):
        return np.frombuffer(text)

    @staticmethod
    def adapt_tuple(tuple):
        return pickle.dumps(tuple)


if __name__ == "__main__":
    import time

    start = time.time()
    db = SQLite()
    data = ((f'/mnt/FOURTH/data/kaggle/faces-data/9326871.{x}.jpg',np.arange(128*32).reshape(128,32)) for x in range(100_000))
    object_made = time.time()

    db.enter_batch_encodings(data)
    done = time.time()

    encodes = db.get_encodes()
    shown = time.time()

    logger.info(
        f"Time to create object: {object_made - start}s\n" +
        f"Time to insert 10_000 rows: {done - object_made}s\n" +
        f"Time to retrieve 10_000 rows: {shown - done}s\n"
    )
