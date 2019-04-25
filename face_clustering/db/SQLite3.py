# imports - standard imports
import sqlite3
import pickle

# imports - third party imports
import numpy as np


class SQLite:
    """
    The commented queries are for the extension
    Current implementation: Bare minimum
    """
    def __init__(self, db_name = ":memory:"):
        self.db_name = db_name
        self.connection = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()

        sqlite3.register_adapter(np.array, SQLite.adapt_array)    
        sqlite3.register_converter("array", SQLite.convert_array)

        sqlite3.register_adapter(tuple, SQLite.adapt_tuple)    #cannot use pickle.dumps directly because of inadequate argument signature 
        sqlite3.register_converter("tuple", pickle.loads)

        # self.cursor.execute(
        #     """CREATE TABLE img (
        #         img_id INTEGER PRIMARY KEY AUTOINCREMENT,
        #         img_path TEXT
        #     )"""
        # )

        # self.cursor.execute(
        #     """CREATE TABLE img_data (
        #         img_id INTEGER,
        #         label_id INTEGER
        #     )"""
        # )

        # self.cursor.execute(
        #     """CREATE TABLE labels (
        #         label_id INTEGER,
        #         label_name TEXT
        #     )"""
        # )
        
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS encodings (
                img_path TEXT,
                encoding array
            )"""
        )

        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS labels (
                img_path TEXT,
                label NUMBER
            )"""
        )

    def __enter__(self):
        return

    def __exit__(self):
        self.cursor.close()
        self.connection.close()

    def enter_label(self, img_path: str, label: int, location_of_face: tuple = None, encoding: np.array = None, time_stamp: object = None):
        # query = "INSERT INTO labels (img_path, loc_box, encoding, time_stamp) VALUES (?, ?, ?, ?)"
        # self.cursor.execute(query, (img_path, location_of_face, encoding, time_stamp))
        self.cursor.execute("INSERT INTO labels (img_path, encoding) VALUES (?, ?)", (img_path, label))
        self.connection.commit()

    def enter_encoding(self, img_path: str, encoding: np.array, location_of_face: tuple = None, time_stamp: object = None):
        # query = "INSERT INTO encodings (img_path, loc_box, encoding, time_stamp) VALUES (?, ?, ?, ?)"
        # self.cursor.execute(query, (img_path, location_of_face, encoding, time_stamp))
        self.cursor.execute("INSERT INTO encodings (img_path, encoding) VALUES (?, ?)", (img_path, encoding))
        self.connection.commit()

    def get_encodes(self):
        self.cursor.execute("SELECT * FROM encodings")
        return self.cursor.fetchall()

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
    db = SQLite("test")
    object_made = time.time()

    for _ in range(100_000):
        db.enter_encoding(
            r'/mnt/FOURTH/data/kaggle/faces-data/9326871.1.jpg', 
                
            # (68, 139, 175, 32), 
                
            np.arange(128*32).reshape(128,32),

            # '2019-04-20 10:00:58.151024'
        )
    done = time.time()
    db.cursor.execute("SELECT * FROM encodings")
    db.cursor.fetchall()
    shown = time.time()

    print(
        f"Time to create object: {object_made - start}s\n" +
        f"Time to insert 100000 rows: {done - object_made}s\n" +
        f"Time to retrieve 100000 rows: {shown - done}s\n"
    )
    