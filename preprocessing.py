# encoding=utf-8
from math import sqrt
import numpy as np
import os
import sqlite3 as sq


class Map:
    name = None
    ranges = None

    def __init__(self, name, ranges):
        self.name = name
        self.ranges = ranges


class GridMap(Map):
    def __init__(self, map__, granularity):
        Map.__init__(self, map__.name, map__.ranges)
        self.grid_ranges = GridMap.get_gridranges(self.ranges, granularity)
        self.granularity = granularity
        self.grid_num = GridMap.get_gridnum(self.grid_ranges)

    @staticmethod
    def get_gridid(grid_lon_num, lon, lat):
        return lon + lat * grid_lon_num

    @staticmethod
    def get_gridranges(ranges, granularity):
        return 0, int((ranges[1] - ranges[0]) / granularity[0]), 0, int((ranges[3] - ranges[2]) / granularity[1])

    @staticmethod
    def get_gridnum(grid_ranges):
        return int((grid_ranges[1] - grid_ranges[0] + 1) * (grid_ranges[3] - grid_ranges[2] + 1))

    @staticmethod
    def get_grid_coordinate(grid_lon_num, grid_id):
        return grid_id % grid_lon_num, int(grid_lon_num / grid_lon_num)

    @staticmethod
    def get_eucdistance(grid_lon_num, grid_id1, grid_id2):
        return sqrt(sum([(a - b) ** 2 for a, b in
                         zip(GridMap.get_grid_coordinate(grid_lon_num, grid_id1),
                             GridMap.get_grid_coordinate(grid_lon_num, grid_id2))]))


class Preprocessing:
    def __init__(self, source_dbfile, source_checkin, source_edges):
        self.source_checkins = source_checkin
        self.source_edges = source_edges
        self.conn = sq.connect(source_dbfile)
        self.cursor = self.conn.cursor()
        self.temp_tables = []
        self.target_checkins = None
        self.target_edges = None

    def start(self):
        """
        Create and initiate a target table to store preprocessed checkins.
        :return: target_table_name: str, a target table to store preprocessed checkins.
        """
        # Create a target table to store preprocessed checkins and copy checkins in the source table into it.
        target_table_name = self.source_checkins + '_'
        sql_copy_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name,
                                  ';CREATE TABLE ', target_table_name,
                                  ' AS SELECT userid, locdatetime, lon, lat, locid',
                                  ' FROM ', self.source_checkins,
                                  ' ORDER BY userid, locdatetime ASC'])
        self.cursor.executescript(sql_copy_table)
        self.temp_tables.append(target_table_name)
        print('Initiate target table: ', target_table_name)
        return target_table_name

    def filter_by_region(self, source_table_name, map__):
        """
        Select checkins within a given geographical range.
        :param source_table_name: str, datasource.
        :param map__: Map, map__.name: name of the region, map__.ranges: 2x2 str list,
        [0][0]: minimum longitude, [0][1]: maximum longitude, [1][0]: minimum latitude, [1][1]: maximum latitude.
        :return: target_table_name: str, checkins filtered by a given region range.
        """
        # Create a target table to store checkin of which the coordinate is within region_range.
        target_table_name = ''.join([source_table_name, map__.name])
        sql_drop_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_table)
        sql_create_table = ''.join(['CREATE TABLE ', target_table_name,
                                    ' AS SELECT userid, locdatetime, lon, lat, locid',
                                    ' FROM ', source_table_name,
                                    ' WHERE lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?'])
        self.cursor.execute(sql_create_table, map__.ranges)

        self.temp_tables.append(target_table_name)
        print('Filtered by region: ', source_table_name, ' --> ', target_table_name)
        return target_table_name

    def filter_by_datetime(self, source_table_name, ranges):
        """
        Select checkins within a given time interval.
        :param source_table_name: str, datasource
        :param ranges: 1x2 str list, [0]: minimum datetime, [1]: maximum datetime,
        :return: target_table_name: str, checkins filtered by a given datetime range.
        """
        # Create a target table.
        target_table_name = ''.join([source_table_name, '_', 'Time'])
        sql_drop_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_table)
        sql_create_table = ''.join(['CREATE TABLE ', target_table_name,
                                    ' AS SELECT userid, locdatetime, lon, lat, locid',
                                    ' FROM ', source_table_name,
                                    ' WHERE locdatetime BETWEEN ? AND ?'])
        self.cursor.execute(sql_create_table, ranges)

        self.temp_tables.append(target_table_name)
        print('Filtered by datetime: ', source_table_name, ' --> ', target_table_name)
        return target_table_name

    def filter_by_tracelength(self, source_table_name, tracelength):
        """
        Select the user of which the trace's length is not less than a given threshold
        :param source_table_name: str, datasource
        :param tracelength: int, minimum length of trace
        :return: target_table_name: str, checkins filtered by a given minimum length of trace.
        """
        # Create a target table to store all records whose
        target_table_name = ''.join([source_table_name, '_Length', str(tracelength)])
        sql_drop_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_table)
        sql_create_table = ''.join(['CREATE TABLE ', target_table_name,
                                    ' AS SELECT userid, locdatetime, lon, lat, locid',
                                    ' FROM ', source_table_name,
                                    ' WHERE userid IN (SELECT userid FROM ', source_table_name,
                                    ' GROUP BY userid HAVING COUNT(*) >= ?)'])
        self.cursor.execute(sql_create_table, (tracelength,))

        self.temp_tables.append(target_table_name)
        print('Filtered by trace length: ', source_table_name, ' --> ', target_table_name)
        return target_table_name

    def subsample(self, source_table_name, interval):
        """
        Subsemple checkins according to a given sampling rate.
        :param source_table_name: str, datasource
        :param interval: int, second(s)
        :return: target_table_name: str, checkins sampled from source table according to a given sampling rate.
        """
        # Get rowid, userid and locdatetime,
        # where the locdatetime is represented by the quotient of seconds since 1970-01-01 00:00:00 and time_interval.
        sql_select_checkins = ''.join(['SELECT rowid, userid,',
                                       ' CAST(STRFTIME(\'%s\', locdatetime) AS INTEGER)/', str(interval),
                                       ' FROM ', source_table_name, ' ORDER BY userid, locdatetime'])
        self.cursor.execute(sql_select_checkins)

        # Take a sample from source table according to the interval.
        sample_rowids = []
        pre_userid = None
        pre_locdatetime = None
        while True:
            results = self.cursor.fetchmany(1000)  # fetchmany() saves much more memory than fetchall()
            if not results:
                break
            for rowid, userid, locdatetime in results:
                # Keep a checkin (rowid) if its userid != previous userid,
                # or its userid = previous userid but its locdatetime != previous locdatetime
                if (userid != pre_userid) or (userid == pre_userid and locdatetime != pre_locdatetime):
                    sample_rowids.append(rowid)
                pre_userid = userid
                pre_locdatetime = locdatetime

        # Create a target table.
        target_table_name = ''.join([source_table_name, '_Sample', str(interval)])
        sql_drop_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_table)
        sql_create_table = ''.join(['CREATE TABLE ', target_table_name,
                                    ' AS SELECT userid, locdatetime, lon, lat, locid',
                                    ' FROM ', source_table_name,
                                    ' WHERE rowid IN ', str(tuple(sample_rowids))])
        self.cursor.execute(sql_create_table)

        self.temp_tables.append(target_table_name)
        print('Subsampling: ', source_table_name, ' --> ', target_table_name)
        return target_table_name

    def clustering(self, source_table_name, k=20, method='mini batch k-means'):
        """
        Cluster the map into clusters.
        :param source_table_name: str, datasource
        :param k: int, number of cluster
        :param method: str, name of clustering method
        :return: target_table_name: str, checkins of which the coordinates are represented by cluster tag.
        """
        # 0. Create a target table by copying the source table and adding a null column for cluster label.
        target_table_name = ''.join([source_table_name, '_Cluster', str(k)])
        sql_create_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name,
                                    ';CREATE TABLE ', target_table_name,
                                    ' AS SELECT userid, locdatetime, lon, lat, locid',
                                    ' FROM ', source_table_name,
                                    ';ALTER TABLE ', target_table_name,
                                    ' ADD COLUMN label INTEGER'])
        self.cursor.executescript(sql_create_table)

        # 1. Begin to cluster.
        # 1.1 Get checkins with corresponding rowids from database.
        sql_select_coordinates = ''.join(['SELECT rowid, lon, lat FROM ', target_table_name])
        # Note that DISTINC shouldn't be used for density-based clustering like DBSCAN.
        self.cursor.execute(sql_select_coordinates)
        rowids, lons, lats = zip(*(self.cursor.fetchall()))
        coordinates = np.column_stack((lons, lats))
        # 1.2 Choose cluster method.
        if method == 'k-means':
            from sklearn.cluster import k_means
            centroid, label, inertia = k_means(coordinates, k)
        elif method == 'mini batch k-means':
            from sklearn.cluster import MiniBatchKMeans
            estimator = MiniBatchKMeans(n_clusters=k, batch_size=k, n_init=10)
            estimator.fit(coordinates)
            label = estimator.labels_
        else:
            raise Exception('Wrong clustering method name.')
        # 1.3 save cluster label to the target table
        sql_update_label = ''.join(['UPDATE ', target_table_name,
                                    ' SET label = ?',
                                    ' WHERE rowid = ?'])
        self.cursor.executemany(sql_update_label, zip(label.tolist(), rowids))

        self.temp_tables.append(target_table_name)
        print('Clustering: ', source_table_name, ' --> ', target_table_name)
        return target_table_name

    def grid(self, source_table_name, gridmap):
        """
        Divide the map into grids.
        :param source_table_name: str, datasource
        :param gridmap: GridMap, gridmap.granularity: 1x2 float list,
        [0]: delta latitude of a grid, [1]: delta longitude of a grid.
        :return: target_table_name: str, checkins of which the coordinates are represented by grid coordinates.
        """
        # Create a target table to store grid coordinates and gridid.
        target_table_name = ''.join([source_table_name, '_', 'Grid'])
        sql_create_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name,
                                    ';CREATE TABLE ', target_table_name,
                                    '(userid INTEGER, locdatetime TEXT, lon INTEGER, lat INTEGER, locid INTEGER,',
                                    ' label INTEGER)'])
        self.cursor.executescript(sql_create_table)
        self.conn.create_function('get_gridid', 3, GridMap.get_gridid)
        sql_create_table = ''.join([
            ' WITH GRID_CHECKINS AS(SELECT userid, locdatetime, locid,',
            ' CAST(((lon - (SELECT MIN(lon) FROM ', source_table_name, ')) / ?) AS INTEGER) AS lon,',
            ' CAST(((lat - (SELECT MIN(lat) FROM ', source_table_name, ')) / ?) AS INTEGER) AS lat',
            ' FROM ', source_table_name,
            ' ORDER BY userid, locdatetime ASC)',
            ' INSERT INTO ', target_table_name,
            ' SELECT userid, locdatetime, lon, lat, locid, get_gridid(?, lon, lat) AS label',
            ' FROM GRID_CHECKINS'])
        self.cursor.execute(sql_create_table,
                            (gridmap.granularity[0], gridmap.granularity[1], gridmap.grid_ranges[1] + 1))

        self.temp_tables.append(target_table_name)
        print('Grid: ', source_table_name, ' --> ', target_table_name)
        return target_table_name

    def create_checkins_index(self, table_name):
        """
        Create index on the table storing preprocessed checkins.
        :param table_name: str, datasource
        :return: nothing
        """
        sql_create_index = ''.join(['DROP INDEX', ' IF EXISTS Index_', table_name,
                                    ';CREATE INDEX Index_', table_name, ' ON ', table_name,
                                    '(userid ASC, locdatetime ASC, lon ASC, lat ASC, locid ASC)'])
        self.cursor.executescript(sql_create_index)
        print('Create index on table: ', table_name)

    def filter_edges(self, source_table_name, target_table_name, checkin_table_name):
        """
        Remove the user whose checkins have been filterred out during the prepreocessing.
        :param source_table_name: str, datasource of social edges.
        :param target_table_name: str, a target table storing filered edges.
        :param checkin_table_name: str, a table storing the users that should be kept.
        :return: nothing
        """
        sql_select_userids = ''.join(['SELECT DISTINCT userid FROM ', checkin_table_name])
        self.cursor.execute(sql_select_userids)
        candidates = [userid[0] for userid in self.cursor.fetchall()]
        candidates_str = str(tuple(candidates))

        adjecent_list = {}
        for candidate in candidates:
            sql_select_friendids = ''.join([
                'SELECT DISTINCT friendid FROM ', source_table_name,
                ' WHERE userid = ? AND friendid IN ', candidates_str])
            self.cursor.execute(sql_select_friendids, (candidate,))
            friendids = set(friendid[0] for friendid in self.cursor.fetchall())
            adjecent_list[candidate] = friendids

        while True:
            invalid_candidate = set()
            for candidate in adjecent_list.keys():
                if not adjecent_list[candidate]:
                    invalid_candidate.add(candidate)
            if not invalid_candidate:
                break
            for candidate in invalid_candidate:
                del adjecent_list[candidate]
            for candidate in adjecent_list.keys():
                adjecent_list[candidate] -= invalid_candidate

        sql_create_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name,
                                    ';CREATE TABLE ', target_table_name, ' (userid INTEGER, friendid INTEGER)'])
        self.cursor.executescript(sql_create_table)
        for candidate in adjecent_list.keys():
            sql_insert_edges = ''.join(['INSERT INTO ', target_table_name, ' (userid, friendid) VALUES (?, ?)'])
            friendids = adjecent_list[candidate]
            self.cursor.executemany(sql_insert_edges, zip([candidate] * len(friendids), friendids))
        # Create index.
        sql_create_index = ''.join(['CREATE INDEX Index_', target_table_name, ' ON ', target_table_name,
                                    '(userid ASC, friendid ASC)'])
        self.cursor.execute(sql_create_index)

        self.target_edges = target_table_name
        print('Filtere dges: ', source_table_name, ' --> ', target_table_name)

    @staticmethod
    def reorder_array(source_array):
        """
        Reorder an array by unique element
        :param source_array: 1xn list
        :return:
        """
        element_dict = dict(zip(set(source_array), range(0, len(source_array))))
        target_array = [element_dict[element] for element in source_array]
        return target_array

    @staticmethod
    def reformat_checkins(source_name, target_name):
        """
        Save source dataset as a new dataset by reordering its column loc_ids.
        :param source_name: str, filename of source dataset.
        :param target_name: str, filename of reformatted dataset.
        :return:
        """
        #  Open file
        with open(source_name, 'r') as source_fid:
            # \r, \n and \r\n will be substituted by \n while reading file if the parameter newline is not specified
            # Load the source checkins to a row_num x 6 list
            data = [row.rstrip('\n').split('\t') for row in source_fid]

            # Delete rows including invalid coordinate.
            invalid_row_index = []
            for row_index in range(len(data) - 1, -1, -1):
                # Traverse data from tha last row.
                latitude, longitude = float(data[row_index][3]), float(data[row_index][4])
                # Invalid coordinate such as lat==0 and lon==0, lat<-80, lat>80, lon<-180, lon>180.
                if (latitude < -80 or latitude > 80) and (longitude < -180 or longitude > 180) or \
                        (latitude == 0 and longitude == 0):
                    invalid_row_index.append(row_index + 1)
                    del data[row_index]

            # Reverse six columns to keep chronological order
            data = data[::-1]

            # Reorder the 5th column loc_id as a int list loc_ids_reordered
            loc_ids_reordered = Preprocessing.reorder_array([row[5] for row in data])

            # Write to a new dataset
            delimiter = '\t'
            with open(target_name, 'w') as target_fid:
                for i in range(0, len(data)):
                    # \n will be substituted by \r, \n or \r\n  while writing file according to your operation system
                    # if the parameter newline is not specified
                    row = delimiter.join([delimiter.join(data[i][0:5]), str(loc_ids_reordered[i])]) + '\n'
                    # If ended in '\r\n', it will be substituted by '\r\r\n'
                    target_fid.write(row)

    @staticmethod
    def split_checkins_by_user(source_name):
        from datetime import datetime

        with open(source_name, 'r') as source_fid:
            user_ids, startends_list, datetimes, locations, locat_ids = [], [], [], [], []

            # Traverse checkins by row to find statting row and ending row of each user.
            start, user_id, user_id_previous, row_index = 0, 0, -1, -1
            for row in source_fid:
                row_index += 1
                elements = row.rstrip('\n').split('\t')

                user_id = int(elements[0])
                if user_id != user_id_previous:
                    # Append (start, end) for the previous user.
                    startends_list.append((user_id_previous, start, row_index - 1))
                    # Set start for current user.
                    start = row_index
                    user_id_previous = user_id
                user_ids.append(user_id)
                datetimes.append(datetime.strptime(elements[1] + ' ' + elements[2], '%Y-%m-%d %H:%M:%S'))
                locations.append((float(elements[3]), float(elements[4])))
                locat_ids.append(int(elements[5]))
            # Delete the first element since it is meaningless.
            del startends_list[0]
            # Append (start, end) for the last user.
            startends_list.append((user_id, start, row_index))

            # Convert each an attribute to a numpy.ndarray except for datetimes.
            user_id_of_last_user = max(user_ids)
            startends = np.zeros((user_id_of_last_user + 1, 2), dtype=np.uint32)
            user_validity = np.zeros(user_id_of_last_user + 1, dtype=np.bool_)
            for row in startends_list:
                startends[row[0], 0] = row[1]
                startends[row[0], 1] = row[2]
                user_validity[row[0]] = True
            locations, locat_ids = np.array(locations), np.array(locat_ids, dtype=np.uint32)

            # Save each attribute as a dat file respectively.
            Preprocessing.save((user_validity, startends, datetimes, locations, locat_ids),
                               (
                                   'user_validity.dat', 'startends.dat', 'datetimes.dat', 'locations.dat',
                                   'locat_ids.dat'))

    @staticmethod
    def save(variables, filenames):
        """
        Serilize variable(s) to current directory according to given filename(s).
        :param variables: 1xn list, variable(s) that need to be saved.
        :param filenames: 1xn list, filename(s) with respective to each variable.
        :return: boolean
        """
        from pickle import dump

        var_num = len(variables)
        if var_num == 0 or var_num != len(filenames):
            return False
        for i in range(0, var_num):
            with open(filenames[i], 'wb') as fid:
                dump(variables[i], fid)
        return True

    @staticmethod
    def load_chekins(attribute_name=None):
        # load serilized variables(checkins) from disk to memory.
        from pickle import load

        if attribute_name == 'USER_VALIDITY':
            with open('user_validity.dat', 'rb') as fid:
                return load(fid)
        elif attribute_name == 'STARTEND':
            with open('startends.dat', 'rb') as fid:
                return load(fid)
        elif attribute_name == 'DATETIME':
            with open('datetimes.dat', 'rb') as fid:
                return load(fid)
        elif attribute_name == 'LOCATION':
            with open('locations.dat', 'rb') as fid:
                return load(fid)
        elif attribute_name == 'LOCAT_ID':
            with open('locat_ids.dat', 'rb') as fid:
                return load(fid)
        else:
            with open('user_validity.dat', 'rb') as validity_fid, \
                    open('startends.dat', 'rb') as startends_fid, \
                    open('datetimes.dat', 'rb') as datetimes_fid, \
                    open('locations.dat', 'rb') as locations_fid, \
                    open('locat_ids.dat', 'rb') as locatids_fid:
                return load(validity_fid), load(startends_fid), load(datetimes_fid), load(locations_fid), load(
                    locatids_fid)

    @staticmethod
    def load_edges(source_name):
        # load serilized variables(edges) from disk to memory.
        from scipy.io import mmread
        from scipy.sparse import dok_matrix

        with open(source_name, 'rb') as fid:
            # Scipy.io.load returns a coo_matrix instead of a dok_matrix.
            return dok_matrix(mmread(fid), dtype=np.int8)

    def stop(self, clean=False, compact=False):
        if clean:
            Preprocessing.clean(self)
        if compact:
            print('Compacting database....')
            self.conn.isolation_level = None  # you shoould set isolation_level as None for python 3.6.0
            self.conn.execute("VACUUM")
        # Save changes and close connection to database.
        self.conn.commit()
        self.conn.close()

    def clean(self):
        """
        Drop temp tables during preprocessing.
        :return: Nothing
        """
        for table_name in self.temp_tables[:(len(self.temp_tables) - 1)]:
            sql_drop_table = ''.join(['DROP TABLE', ' IF EXISTS ', table_name])
            self.cursor.execute(sql_drop_table)
        self.target_checkins = self.temp_tables[-1]


def main():
    # 0. choose a dataset
    # os.chdir('D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Brightkite')
    os.chdir('D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Gowalla')

    p = Preprocessing('checkins.db', 'Checkins', 'Edges')

    # # 0. dump data in csv files into database.
    # p.reformat_checkins('checkins.txt', 'checkins_valid_reordered.txt')
    # p.split_checkins_by_user('checkins_valid_reordered.txt')
    # uservalidity, startend, datetime, location, locatid = p.Preprocessing.load_chekins()
    # edge = p.load_edges('edges_preview_in_sparse_matrix.dat')

    # 1. checkins preprocessing.
    target_checkins = p.start()
    target_checkins = p.filter_by_region(target_checkins, map_)
    target_checkins = p.filter_by_datetime(target_checkins, datetime_range)
    target_checkins = p.subsample(target_checkins, subsample_interval)
    target_checkins = p.filter_by_tracelength(target_checkins, min_tracelength)
    target_checkins = p.clustering(target_checkins, k=cluster_num)
    # target_checkins = p.grid(target_checkins, grid_map)
    p.create_checkins_index(target_checkins)

    # 2. edges preprocessing
    target_edges = p.source_edges + target_checkins.replace(p.source_checkins, '')
    p.filter_edges(p.source_edges, target_edges, target_checkins)

    # 3. Clean temp tables, compact the database file and close connection to database.
    p.stop(clean=True, compact=False)


map_austin = Map('Austin', (-97.7714033167, -97.5977249833, 30.19719445, 30.4448463144))
map_sf = Map('SF', (-122.521368, -122.356684, 37.706357, 37.817344))
map_sto = Map('Stockholm', (17.911736377, 18.088630197, 59.1932443, 59.4409599167))

# San Francisco in Brightkite with users whose trajectory's length >= 50.
map_ = map_sto
cluster_num = 50
grid_map = GridMap(map_, (0.0055625, 0.00444375))  # 500 meters x 500 meters
datetime_range = ('2008-03-21 20:36:21', '2010-10-23 05:22:06')
min_tracelength = 50
subsample_interval = 1 * 60 * 60  # in seconds

if __name__ == '__main__':
    main()
