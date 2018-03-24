# encoding=utf-8
import sqlite3 as sq


class DBManipulation:
    def __init__(self, dataset):
        self.conn = sq.connect(dataset.dbfile, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def vacuum(self):
        self.conn.isolation_level = None  # you shoould set isolation_level as None for python 3.6+
        self.cursor.execute('VACUUM')
