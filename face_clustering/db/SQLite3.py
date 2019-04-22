# imports - standard imports
import sqlite3
import io

# imports - third party imports
import numpy as np


class SQLite:
    def __init__(self):
        self.db_name = ":memory:"
        self.connection = sqlite3.connect(self.db_name, detect_types=sqlite3.PARSE_DECLTYPES)

        sqlite3.register_adapter(np.ndarray, SQLite.adapt_array)    
        sqlite3.register_converter("array", SQLite.convert_array)

    
    def close(self):
        pass

    @staticmethod
    def adapt_array(arr):
        # REF: http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    @staticmethod
    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)


if __name__ == "__main__":
    x = np.arange(12).reshape(2,6)
    inst = SQLite()
    cur = inst.connection.cursor()
    cur.execute("create table test (arr array)")
    cur.execute("desc test")
    cur.close()
