# encoding=utf-8
import sqlite3 as sq
from math import sqrt

import numpy as np
from dataset import SNAPGowalla, SNAPGowallaAustin


class GridMap:
    def __init__(self, dataset, granularity):
        self.name = dataset.name
        self.ranges = dataset.ranges
        self.granularity = granularity
        self.grid_ranges = self.get_gridranges()
        self.grid_num = self.get_gridnum()

    def get_gridranges(self):
        return 0, \
               int((self.ranges[1] - self.ranges[0]) / self.granularity[0]), \
               0, \
               int((self.ranges[3] - self.ranges[2]) / self.granularity[1])

    def get_gridnum(self):
        return int((self.grid_ranges[1] - self.grid_ranges[0] + 1) * (self.grid_ranges[3] - self.grid_ranges[2] + 1))

    @staticmethod
    def get_gridid(grid_lon_num, lon, lat):
        return lon + lat * grid_lon_num

    @staticmethod
    def get_grid_coordinate(grid_lon_num, grid_id):
        return grid_id % grid_lon_num, int(grid_lon_num / grid_lon_num)

    @staticmethod
    def get_eucdistance(grid_lon_num, grid_id1, grid_id2):
        return sqrt(sum([(a - b) ** 2 for a, b in
                         zip(GridMap.get_grid_coordinate(grid_lon_num, grid_id1),
                             GridMap.get_grid_coordinate(grid_lon_num, grid_id2))]))


class Preprocessing:
    def __init__(self, dataset):
        self.source_checkins = dataset.checkins
        self.source_edges = dataset.edges
        self.conn = sq.connect(dataset.url)
        self.cursor = self.conn.cursor()
        self.temp_tables = []
        self.target_checkins = None
        self.target_edges = None

    def start(self):
        """
        Create and initiate a target table to store preprocessed checkins.
        :return: target_table_name: str, a target table to store preprocessed checkins.
        """
        # Create a target table by copying source checkins.
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

    def filter_by_region(self, source_table_name, region):
        """
        Select checkins within a given geographical range.
        :param source_table_name: str, datasource.
        :param region: Dataset, name: name of the region, ranges: 1x4 float list,
        [0]: minimum longitude, [1]: maximum longitude, [2]: minimum latitude, [3]: maximum latitude.
        :return: target_table_name: str, checkins filtered by a given region range.
        """
        # Create a target table to store checkins of which the coordinates are within region_.
        target_table_name = ''.join([source_table_name, region.name])
        sql_drop_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_table)
        sql_create_table = ''.join(['CREATE TABLE ', target_table_name,
                                    ' AS SELECT userid, locdatetime, lon, lat, locid',
                                    ' FROM ', source_table_name,
                                    ' WHERE lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?'])
        self.cursor.execute(sql_create_table, region.ranges)

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
        Select any user of which the trace's length is not less than a given threshold
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
        Subsample checkins according to a given sampling rate.
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
        :return: target_table_name: str, checkins of which the coordinates are represented by cluster centers.
        """
        # 0. Create a target table by copying the source table and adding a null column for cluster label.
        target_table_name = ''.join([source_table_name, '_Cluster', str(k)])
        sql_create_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name,
                                    ';CREATE TABLE ', target_table_name,
                                    ' AS SELECT userid, locdatetime, lon, lat, locid',
                                    ' FROM ', source_table_name])
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
            cluster_centers, label, inertia = k_means(coordinates, k)
        elif method == 'mini batch k-means':
            from sklearn.cluster import MiniBatchKMeans
            estimator = MiniBatchKMeans(n_clusters=k, batch_size=k, n_init=10)
            estimator.fit(coordinates)
            cluster_centers = [estimator.cluster_centers_[label] for label in estimator.labels_]
        else:
            raise Exception('Wrong clustering method name.')
        cluster_center_lons, cluster_center_lats = zip(*cluster_centers)

        # 2 save cluster center coordinates to the target table, 
        # i.e. replace original coordinates with its cluster center's coordinate.
        sql_update_label = ''.join(['UPDATE ', target_table_name,
                                    ' SET lon = ?, lat = ?',
                                    ' WHERE rowid = ?'])
        self.cursor.executemany(sql_update_label, zip(cluster_center_lons, cluster_center_lats, rowids))

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
        sql_drop_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_table)
        sql_create_table = ''.join([
            'CREATE TABLE ', target_table_name, ' AS',
            ' SELECT userid, locdatetime, locid,',
            ' CAST(((lon - (SELECT MIN(lon) FROM ', source_table_name, ')) / ?) AS INTEGER) AS lon,',
            ' CAST(((lat - (SELECT MIN(lat) FROM ', source_table_name, ')) / ?) AS INTEGER) AS lat',
            ' FROM ', source_table_name,
            ' ORDER BY userid, locdatetime ASC'])
        self.cursor.execute(sql_create_table, (gridmap.granularity[0], gridmap.granularity[1]))

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
        Remove any user whose checkins have been filterred out.
        :param source_table_name: str, datasource of social edges.
        :param target_table_name: str, a target table storing filered edges.
        :param checkin_table_name: str, a table storing the users that should be kept.
        :return: nothing
        """
        sql_create_table = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name,
                                    ';CREATE TABLE ', target_table_name, ' AS',
                                    ' SELECT userid as userid, friendid as friendid',
                                    ' FROM ', source_table_name,
                                    ' WHERE userid IN ',
                                    '(SELECT DISTINCT userid',
                                    ' FROM ', checkin_table_name, ')'])
        self.cursor.executescript(sql_create_table)

        while True:
            sql_delete = ''.join(['DELETE FROM ', target_table_name,
                                  ' WHERE friendid NOT IN ',
                                  '(SELECT DISTINCT userid',
                                  ' FROM ', target_table_name, ')'])
            self.cursor.execute(sql_delete)
            if self.cursor.rowcount == 0:
                break

        sql_delete = ''.join(['DELETE FROM ', checkin_table_name,
                              ' WHERE userid NOT IN',
                              '(SELECT DISTINCT userid',
                              ' FROM ', target_table_name, ')'])
        self.cursor.execute(sql_delete)

        self.target_edges = target_table_name
        print('Filtere dges: ', source_table_name, ' --> ', target_table_name)

    def create_edges_index(self, table_name):
        """
        Create index on the table storing preprocessed edges.
        :param table_name: str, datasource
        :return: nothing
        """
        sql_create_index = ''.join(['CREATE INDEX Index_', table_name, ' ON ', table_name,
                                    '(userid ASC, friendid ASC)'])
        self.cursor.execute(sql_create_index)

    def stop(self, clean=True, compact=False):
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
        Drop temp tables.
        :return: Nothing
        """
        for table_name in self.temp_tables[:(len(self.temp_tables) - 1)]:
            sql_drop_table = ''.join(['DROP TABLE', ' IF EXISTS ', table_name])
            self.cursor.execute(sql_drop_table)
        self.target_checkins = self.temp_tables[-1]


def preprocessing():
    # 0. choose a dataset and set filtering parameters
    dataset = SNAPGowalla
    filter_region = SNAPGowallaAustin
    cluster_num = 50
    # grid_map = GridMap(filter_region, (0.0055625, 0.00444375))  # 500 meters x 500 meters
    datetime_range = ('2008-03-21 20:36:21', '2010-10-23 05:22:06')
    min_tracelength = 50
    subsample_interval = 1 * 60 * 60  # in seconds

    p = Preprocessing(dataset)

    # 1. checkins preprocessing.
    target_checkins = p.start()
    target_checkins = p.filter_by_region(target_checkins, filter_region)
    target_checkins = p.filter_by_datetime(target_checkins, datetime_range)
    target_checkins = p.subsample(target_checkins, subsample_interval)
    target_checkins = p.filter_by_tracelength(target_checkins, min_tracelength)
    target_checkins = p.clustering(target_checkins, k=cluster_num)
    # target_checkins = p.grid(target_checkins, grid_map)
    p.create_checkins_index(target_checkins)

    # 2. edges preprocessing
    target_edges = p.source_edges + target_checkins.replace(p.source_checkins, '')
    p.filter_edges(p.source_edges, target_edges, target_checkins)
    p.create_edges_index(target_edges)

    # 3. Clean temp tables, compact the database file and close connection to database.
    p.stop(clean=True, compact=False)


if __name__ == '__main__':
    preprocessing()
