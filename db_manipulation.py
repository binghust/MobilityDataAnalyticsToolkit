# encoding=utf-8
import sqlite3 as sq


class DBManipulation:
    dbfiles = {
        'Brightkite': 'D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Brightkite\\checkins.db',
        'Gowalla': 'D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Gowalla\\checkins.db'
    }

    def __init__(self, dataset_name):
        if not isinstance(dataset_name, str):
            raise TypeError
        if dataset_name not in DBManipulation.dbfiles:
            raise ValueError
        dbfile = DBManipulation.dbfiles[dataset_name]
        self.conn = sq.connect(dbfile, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def vacuum(self):
        self.conn.isolation_level = None  # you shoould set isolation_level as None for python 3.6+
        self.cursor.execute('VACUUM')
