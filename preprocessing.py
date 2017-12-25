# encoding=utf-8
import numpy as np
import sqlite3 as sq


class Map:
    name = None
    ranges = None

    def __init__(self, name, ranges):
        self.name = name
        self.ranges = ranges


class GridMap(Map):
    grid_ranges = None
    granularity = None
    grid_num = None

    def __init__(self, map__, granularity):
        Map.__init__(self, map__.name, map__.ranges)
        self.grid_ranges = GridMap.get_gridranges(self.ranges, granularity)
        self.granularity = granularity
        self.grid_num = GridMap.get_gridnum(self.grid_ranges)

    def get_gridid(self, grid_coordinate):
        return grid_coordinate[0] + grid_coordinate[1] * self.grid_ranges[1]

    @staticmethod
    def get_gridranges(ranges, granularity):
        return 0, int((ranges[1] - ranges[0]) / granularity[0]), 0, int((ranges[3] - ranges[2]) / granularity[1])

    @staticmethod
    def get_gridnum(grid_ranges):
        return int((grid_ranges[1] - grid_ranges[0]) * (grid_ranges[3] - grid_ranges[2]))


class Preprocessing:
    source_checkins = None
    source_edges = None
    conn = None
    cursor = None
    temp_tables = []
    target_checkins = None
    target_edges = None

    def __init__(self, source_dbfile, source_checkin, source_edges):
        self.source_checkins = source_checkin
        self.source_edges = source_edges
        Preprocessing.connect2db(self, source_dbfile)

    def connect2db(self, source_dbfile):
        # Connect to sqlite database. Database file should be placed under current directory.
        self.conn = sq.connect(source_dbfile)
        self.cursor = self.conn.cursor()

    def start(self):
        """
        Create and initiate a target table to store preprocessed checkins.
        :return: target_table_name: str, a target table to store preprocessed checkins.
        """
        # Create a target table to store preprocessed checkins.
        target_table_name = self.source_checkins + '_'
        sql_drop_oldtargettable = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_oldtargettable)
        sql_create_targettable = ''.join(['CREATE TABLE ', target_table_name,
                                          ' (userid INTEGER, locdatetime TEXT, lon REAL, lat REAL, locid INTEGER)'])
        self.cursor.execute(sql_create_targettable)

        # Copy checkins in the source table into the new table.
        # All preprocessions followed will be done on the new table.
        sql_copyall = ''.join(['INSERT INTO ', target_table_name,
                               ' SELECT userid, locdatetime, lon, lat, locid FROM ', self.source_checkins,
                               ' ORDER BY userid, locdatetime ASC'])
        self.cursor.execute(sql_copyall)

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
        # Create a target table.
        target_table_name = ''.join([source_table_name, map__.name])
        sql_drop_oldtargetable = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_oldtargetable)
        sql_create_targettable = ''.join(['CREATE TABLE ', target_table_name,
                                          ' (userid INTEGER, locdatetime TEXT, lon REAL, lat REAL, locid INTEGER)'])
        self.cursor.execute(sql_create_targettable)

        # Insert checkin of which the coordinate is within region_range into the target table.
        sql_filter = ''.join(['INSERT INTO ', target_table_name, ' SELECT * FROM ', source_table_name,
                              ' WHERE lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?'])
        self.cursor.execute(sql_filter, map__.ranges)

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
        sql_drop_oldtargettable = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_oldtargettable)
        sql_create_targettable = ''.join(['CREATE TABLE ', target_table_name,
                                          '(userid INTEGER, locdatetime TEXT, lon REAL, lat REAL, locid INTEGER)'])
        self.cursor.execute(sql_create_targettable)

        # Insert the checkin of which the locdatetime is within datatime_range into the target table.
        sql_filter = ''.join(['INSERT INTO ', target_table_name, ' SELECT * FROM ', source_table_name,
                              ' WHERE locdatetime BETWEEN \'', ranges[0], '\' AND \'', ranges[1], '\''])
        self.cursor.execute(sql_filter)

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
        target_table_name = ''.join([source_table_name, '_', str(tracelength)])
        sql_drop_oldtargettable = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_oldtargettable)
        sql_create_targettable = ''.join(['CREATE TABLE ', target_table_name,
                                          '(userid INTEGER, locdatetime TEXT, lon REAL, lat REAL, locid INTEGER)'])
        self.cursor.execute(sql_create_targettable)

        # Insert the checkin of which the length is >= tracelength into the target table.
        sql_filter = ''.join(['INSERT INTO ', target_table_name, ' SELECT * FROM ', source_table_name,
                              ' WHERE userid IN (SELECT userid FROM ', source_table_name,
                              ' GROUP BY userid HAVING COUNT(*) >= ?)'])
        self.cursor.execute(sql_filter, (tracelength,))

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
        sql_select_all_checkins = ''.join(['SELECT rowid, userid,',
                                           ' CAST(STRFTIME(\'%s\', locdatetime) AS INTEGER)/', str(interval),
                                           ' FROM ', source_table_name, ' ORDER BY userid, locdatetime'])
        self.cursor.execute(sql_select_all_checkins)
        results = self.cursor.fetchall()

        # Take a sample from source table according to the interval.
        sample_rowids = []
        pre_userid = None
        pre_locdatetime = None
        for rowid, userid, locdatetime in results:
            # Keep a checkin (rowid) if its userid != previous userid,
            # or its userid = previous userid but its locdatetime != previous locdatetime
            if (userid != pre_userid) or (userid == pre_userid and locdatetime != pre_locdatetime):
                sample_rowids.append(rowid)
            pre_userid = userid
            pre_locdatetime = locdatetime

        # Create a target table.
        target_table_name = ''.join([source_table_name, '_', str(interval)])
        sql_drop_oldtargettable = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_oldtargettable)
        sql_create_targettable = ''.join(['CREATE TABLE ', target_table_name,
                                          '(userid INTEGER, locdatetime TEXT, lon REAL, lat REAL, locid INTEGER)'])
        self.cursor.execute(sql_create_targettable)

        # Insert sampled checkins into the target table.
        sql_insert_temptable = ''.join(['INSERT INTO ', target_table_name, ' SELECT * FROM ', source_table_name,
                                        ' WHERE rowid IN ', str(tuple(sample_rowids))])
        self.cursor.execute(sql_insert_temptable)

        self.temp_tables.append(target_table_name)
        print('Subsampling: ', source_table_name, ' --> ', target_table_name)
        return target_table_name

    def discretization(self, source_table_name, gridmap):
        """
        Divide the map into grids.
        :param source_table_name: str, datasource
        :param gridmap: GridMap, gridmap.granularity: 1x2 float list,
        [0]: delta latitude of a grid, [1]: delta longitude of a grid.
        :return: target_table_name: str, checkins of which the coordinates are represented by grid coordinates.
        """
        # Get all source checkins.
        sql_getall = ''.join(['SELECT userid, locdatetime, lon, lat, locid FROM ', source_table_name,
                              ' ORDER BY userid, locdatetime'])
        self.cursor.execute(sql_getall)
        old_checkins = self.cursor.fetchall()

        # Compute grid coordinates for all source checkins.
        lons = [lon for userid, locdatetime, lon, lat, locid in old_checkins]
        lats = [lat for userid, locdatetime, lon, lat, locid in old_checkins]
        min_lon = min(lons)
        min_lat = min(lats)
        new_checkins = [(userid, locdatetime,
                         int((lon - min_lon) / gridmap.granularity[0]),
                         int((lat - min_lat) / gridmap.granularity[1]),
                         locid) for userid, locdatetime, lon, lat, locid in old_checkins]

        # Create a target table.
        target_table_name = ''.join([source_table_name, '_', 'Grid'])
        sql_drop_oldtargettable = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_oldtargettable)
        sql_create_targettable = ''.join(
            ['CREATE TABLE ', target_table_name,
             ' (userid INTEGER, locdatetime TEXT, gridlon INTEGER, gridlat INTEGER, locid INTEGER)'])
        self.cursor.execute(sql_create_targettable)

        # Insert into the target table the checkins containing grid coordinates.
        sql_insert_checkins = ''.join(['INSERT INTO ', target_table_name, ' VALUES (?, ?, ?, ?, ?)'])
        self.cursor.executemany(sql_insert_checkins, new_checkins)

        self.temp_tables.append(target_table_name)
        print('Discretization: ', source_table_name, ' --> ', target_table_name)
        return target_table_name

    def create_checkins_index(self, table_name):
        """
        Create index on the table storing preprocessed checkins.
        :param table_name: str, datasource
        :return: nothing
        """
        sql_drop_oldindex = ''.join(['DROP INDEX', ' IF EXISTS Index_', table_name])
        self.cursor.execute(sql_drop_oldindex)
        sql_create_index = ''.join(['CREATE INDEX Index_', table_name, ' ON ', table_name,
                                    '(userid ASC, locdatetime ASC, gridlon ASC, gridlat ASC, locid ASC)'])
        self.cursor.execute(sql_create_index)
        print('Create index on table: ', table_name)

    def filter_edges(self, source_table_name, target_table_name, checkin_table_name):
        """
        Remove the user whose checkins have been filterred out during the prepreocessing.
        :param source_table_name: str, datasource of social edges.
        :param target_table_name: str, a target table storing filered edges.
        :param checkin_table_name: str, a table storing the users that should be kept.
        :return: nothing
        """
        # Create a target table.
        sql_drop_oldtargettable = ''.join(['DROP TABLE', ' IF EXISTS ', target_table_name])
        self.cursor.execute(sql_drop_oldtargettable)
        sql_create_targettable = ''.join(['CREATE TABLE ', target_table_name, '(userid INTEGER, friendid INTEGER)'])
        self.cursor.execute(sql_create_targettable)

        # Remove the user from userids and friends.
        sql_copyall = ''.join(['INSERT INTO ', target_table_name, ' SELECT userid, friendid FROM ', source_table_name])
        self.cursor.execute(sql_copyall)
        sql_delete_user = ''.join(['DELETE FROM ', target_table_name,
                                   ' WHERE userid NOT IN (SELECT userid FROM ', checkin_table_name, ')'])
        self.cursor.execute(sql_delete_user)
        sql_delete_friend = ''.join(['DELETE FROM ', target_table_name,
                                     ' WHERE friendid NOT IN (SELECT userid FROM ', checkin_table_name, ')'])
        self.cursor.execute(sql_delete_friend)

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


# San Francisco in Brightkite with users whose trajectory's length >= 50.
map_ = Map('SF', (-122.521368, -122.356684, 37.706357, 37.817344))
grid_map = GridMap(map_, (0.0055625, 0.00444375))  # 500 meters x 500 meters
datetime_range = ('2008-03-21 20:36:21', '2010-10-23 05:22:06')
min_tracelength = 50
subsample_interval = 1 * 60 * 60  # seconds
