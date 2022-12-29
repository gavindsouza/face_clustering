"""
Using psql
try using sqlite3
"""
# imports - standard imports
import os
import multiprocessing
import logging

# imports - third party imports
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


logger = logging.getLogger(__file__)


if __name__ == '__main__':
    # setting up db conn
    db_name = "face_cluster"

    try:
        conn = psycopg2.connect(database=db_name, user="postgres", password="")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        logger.info(f"'{db_name}' db connected!")

    except psycopg2.OperationalError as Message:
        logger.info(Message)
        logger.info("Database doesn't exist\nCreating database...")

        with psycopg2.connect(database="postgres", user="postgres", password="") as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE {db_name};")

        conn = psycopg2.connect(database=db_name, user="postgres", password="")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        logger.info(f"'{db_name}' db connected!")

    #
    # beech ka content
    #

    cur.close()
    conn.close()
