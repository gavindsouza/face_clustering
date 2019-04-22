# imports - standard imports
import sqlite3
import pickle

# imports - third party imports
import numpy as np


class SQLite:
    def __init__(self, db_name = ":memory:"):
        self.db_name = db_name
        self.connection = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.connection.cursor()

        sqlite3.register_adapter(np.array, SQLite.adapt_array)    
        sqlite3.register_converter("array", SQLite.convert_array)

        sqlite3.register_adapter(tuple, SQLite.adapt_tuple)    #cannot use pickle.dumps directly because of inadequate argument signature 
        sqlite3.register_converter("tuple", pickle.loads)

        self.cursor.execute(
            """CREATE TABLE main_entry (
                img_path TEXT,
                loc_box tuple,
                encoding array,
                time_stamp TEXT
            )"""
        )

    def __enter__(self):
        return

    def __exit__(self):
        self.cursor.close()
        self.connection.close()

    def entry(self, img_path: str, location_of_face: tuple, encoding: np.array, time_stamp: object):
        query = f"INSERT INTO main_entry (img_path, loc_box, encoding, time_stamp) VALUES (?, ?, ?, ?)"
        self.cursor.execute(query, (img_path, location_of_face, encoding, time_stamp))
        self.connection.commit()

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
    x = np.arange(12).reshape(2,6)
    db = SQLite()
    db.entry(
        r'/mnt/FOURTH/data/kaggle/faces-data/9326871.1.jpg', 
            
        (68, 139, 175, 32), 
            
        np.array([-1.26975790e-01,  4.97269630e-02,  7.74217919e-02, -2.01154444e-02,
            -4.73027192e-02,  9.92646255e-03, -1.63344622e-01, -9.74124447e-02,
            7.63679072e-02,  1.44913550e-02,  4.76824678e-02,  8.57453421e-02]),
            
        '2019-04-20 10:00:58.151024'
    )
    db.cursor.execute("SELECT * FROM main_entry")
    print(db.cursor.fetchall())
    