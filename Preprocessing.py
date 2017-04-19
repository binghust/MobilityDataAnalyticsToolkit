import matplotlib.pyplot as plt
import numpy as np
import sqlite3 as sq
from enum import Enum
from scipy.sparse import dok_matrix


class Dataset(Enum):
    Brighkite = r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Brightkite'
    Gowalla = r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Gowalla'


# Choose a dataset before preprocessing and set its directory as current directory.
def choose_dataset(dataset_name):
    from os import chdir

    if dataset_name == 'Gowalla':
        chdir(Dataset.Gowalla.value)
        return True
    elif dataset_name == 'Brightkite':
        chdir(Dataset.Brighkite.value)
        return True
    else:
        return False


# Reorder an array
def reorder_array(source_array):
    element_dict = dict(zip(set(source_array), range(0, len(source_array))))
    target_array = [element_dict[element] for element in source_array]
    return target_array


# Serilize variable(s) to current directory according to given filename(s).
def save(variable, filename):
    from pickle import dump

    var_num = len(variable)
    if var_num == 0 or var_num != len(filename):
        return False
    for i in range(0, var_num):
        with open(filename[i], 'wb') as fid:
            dump(variable[i], fid)
    return True


# Save original dataset as a new dataset by reordering its column loc_ids
def reformat_checkins(source_name, target_name):
    #  Open file
    with open(source_name, 'r') as source_fid:
        # \r, \n and \r\n will be substituted by \n while reading file if the parameter newline is not specified
        # Load the original checkins to a row_num x 6 list
        data = [row.rstrip('\n').split('\t') for row in source_fid]

        # Delete rows including invalid coordinate.
        invalid_row_index = []
        for row_index in range(len(data) - 1, -1, -1):
            # Traverse data from tha last row.
            latitude, longitude = float(data[row_index][3]), float(data[row_index][4])
            # Invalid coordinate such as lat==0 and lon==0, lat<-80, lat>80, lon<-180, lon>180.
            if (latitude < -80 or latitude > 80) and (longitude < -180 or longitude > 180) or\
                    (latitude == 0 and longitude == 0):
                invalid_row_index.append(row_index + 1)
                del data[row_index]

        # Reverse six columns to keep chronological order
        data = data[::-1]

        # Reorder the 5th column loc_id as a int list loc_ids_reordered
        loc_ids_reordered = reorder_array([row[5] for row in data])

        # Write to a new dataset
        delimiter = '\t'
        with open(target_name, 'w') as target_fid:
            for i in range(0, len(data)):
                # \n will be substituted by \r, \n or \r\n  while writing file according to your operation system
                # if the parameter newline is not specified
                row = delimiter.join([delimiter.join(data[i][0:5]), str(loc_ids_reordered[i])]) + '\n'
                # If ended in '\r\n', it will be substituted by '\r\r\n'
                target_fid.write(row)


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
        save((user_validity, startends, datetimes, locations, locat_ids),
             ('user_validity.dat', 'startends.dat', 'datetimes.dat', 'locations.dat', 'locat_ids.dat'))


def reformat_edges(source_name, target_name):
    # Can not use scipy.io.savemat since sparse matrix is unhashable.
    from scipy.io import mmwrite

    with open(source_name, 'r') as source_fid:
        # Load the original edges to a row_num x 2 list
        data = [row.rstrip('\n').split('\t') for row in source_fid]

        # Create a sparse matrix in which an element assigned 1 represents the existence of friendship.
        user_num = int(data[len(data) - 1][0]) + 1
        matrix = dok_matrix((user_num, user_num), dtype=np.int8)
        for row_index in range(0, len(data)):
            matrix[int(data[row_index][0]), int(data[row_index][1])] = 1

        # Save the sparse matrix as a file named target_name in current directory.
        with open(target_name, 'wb') as target_fid:
            # Use scipy.io.mmwrite instead of pickle.dump since the latter can't deal with a sparse matrix.
            mmwrite(target_fid, matrix)


def load_chekins(attribute_name=None):
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
            return load(validity_fid), load(startends_fid), load(datetimes_fid), load(locations_fid), load(locatids_fid)


def load_edges(source_name):
    from scipy.io import mmread

    with open(source_name, 'rb') as fid:
        # Scipy.io.load returns a coo_matrix instead of a dok_matrix.
        return dok_matrix(mmread(fid), dtype=np.int8)


def plot_in_geoplotlib(region_range):
    import geoplotlib as gp

    # Connect to sqlite database. Database file should be placed in current directory.
    conn = sq.connect('checkins.db')
    cursor = conn.cursor()

    # Get locations within region_range.
    cursor.execute('SELECT longitude, latitude FROM Checkins_Preview '
                   'WHERE longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?',
                   (region_range[0][0], region_range[0][1], region_range[1][0], region_range[1][1]))
    locations = cursor.fetchall()
    # Use numpy.ndarray instead of list, otherwise it failed.
    longitudes = np.array([location[0] for location in locations])
    latitudes = np.array([location[1] for location in locations])

    gp.hist(gp.utils.DataAccessObject({'lon': longitudes, 'lat': latitudes}), colorscale='sqrt', binsize=8)
    gp.show()

    # Close the connection to database.
    cursor.close()
    conn.close()


def checkin_num_distribution(attribute_name, order=None, order_by_count=None):
    # Connect to sqlite database. Database file should be placed under current directory.
    conn = sq.connect('checkins.db')
    cursor = conn.cursor()

    # Get total number of checkins.
    cursor.execute('SELECT count(?) FROM Checkins', (attribute_name, ))
    checkin_total_num = cursor.fetchall()[0][0]

    # Get attibutes of all of users.
    if order:
        if order_by_count:
            cursor.execute('SELECT ' + attribute_name + ', count(*) FROM Checkins GROUP BY ' +
                           attribute_name + ' ORDER BY count(*) ' + order)
        else:
            cursor.execute('SELECT ' + attribute_name + ', count(*) FROM Checkins GROUP BY ' +
                           attribute_name + ' ORDER BY ' + attribute_name + ' ' + order)
    else:
        cursor.execute('SELECT ' + attribute_name + ', count(*) FROM Checkins GROUP BY ' + attribute_name)
    result = cursor.fetchall()
    checkin_num_by_attribute = np.array([num[1] for num in result])

    # Calulate cumulative_distribution of checkin number.
    checkin_count = 0
    cumulative_distribution = np.empty(checkin_num_by_attribute.size, dtype=np.float16)
    for i in range(0, checkin_num_by_attribute.size):
        checkin_count += checkin_num_by_attribute[i]
        cumulative_distribution[i] = checkin_count / checkin_total_num

    # Plot checkin number and its cumulative_distribution.
    xaxis = np.linspace(1, checkin_num_by_attribute.size, checkin_num_by_attribute.size, dtype=np.uint32)
    plt.plot(xaxis, cumulative_distribution)
    plt.twinx()
    plt.scatter(xaxis, checkin_num_by_attribute)
    plt.show()

    # Close connection to database.
    cursor.close()
    conn.close()


def plot_locations_by_user():
    # Connect to sqlite database. Database file should be placed under current directory.
    conn = sq.connect('checkins.db')
    cursor = conn.cursor()

    # Get user_ids of all of users.
    cursor.execute('SELECT DISTINCT userid FROM Checkins_Preview')
    user_ids = cursor.fetchall()

    # Generate a color for each user's plot
    colormap = plt.get_cmap('gnuplot')
    colors = [colormap(i) for i in np.linspace(0, 1, len(user_ids))]

    for user_id, color in zip(user_ids, colors):
        # Get locations of each user.
        cursor.execute('SELECT longitude, latitude FROM Checkins_Preview WHERE userid = ?', (user_id[0],))
        locations = np.array(cursor.fetchall())
        # Plot locations of each user by regarding longitude as X-axis, latitude as Y-axis.
        # Plotting costs much more time than querying database does.
        plt.scatter(locations[:, 0], locations[:, 1], c=color, marker='o')
    plt.show()

    # Close the connection to database.
    cursor.close()
    conn.close()


def plot_locations_by_region(region_range):
    # Connect to sqlite database. Database file should be placed under current directory.
    conn = sq.connect('checkins.db')
    cursor = conn.cursor()

    # Get total number of checkins.
    cursor.execute('SELECT longitude, latitude FROM Checkins_Preview '
                   'WHERE longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?',
                   (region_range[0][0], region_range[0][1], region_range[1][0], region_range[1][1]))
    locations = cursor.fetchall()
    longitudes = np.array([location[0] for location in locations])
    latitudes = np.array([location[1] for location in locations])
    plt.scatter(longitudes, latitudes, c='b', marker='o')
    plt.show()

    # Close the connection to database.
    cursor.close()
    conn.close()


def generate_rectangle():
    from matplotlib.patches import Rectangle

    # Connect to sqlite database. Database file should be placed under current directory.
    conn = sq.connect('checkins.db')
    cursor = conn.cursor()

    # Get user_ids of all of users.
    cursor.execute('SELECT latitude, longitude FROM Checkins_Preview')
    locations = cursor.fetchall()
    latitudes = np.array([location[0] for location in locations])
    longitudes = np.array([location[1] for location in locations])

    point_num_in_rec = 4
    rectangles = []
    for i in range(0, len(locations), point_num_in_rec):
        top = max(latitudes[i:i + point_num_in_rec])
        bottom = min(latitudes[i:i + point_num_in_rec])
        left = min(longitudes[i:i + point_num_in_rec])
        right = max(longitudes[i:i + point_num_in_rec])
        rectangle = Rectangle((left, bottom), right - left, top - bottom, fill=False)
        rectangles.append(rectangle)

    # Close the connection to database.
    cursor.close()
    conn.close()

    return rectangles


def plot_rectangles():
    fig = plt.figure(1)
    axes = fig.add_subplot(111)
    rectangles = generate_rectangle()
    for rectangle in rectangles:
        axes.add_patch(rectangle)
    plt.plot(-122.419415, 37.774929)
    plt.show()


def plot_trajectory_by_user(userid):
    # Connect to sqlite database. Database file should be placed under current directory.
    conn = sq.connect('checkins.db')
    cursor = conn.cursor()

    # Get total number of checkins.
    cursor.execute('SELECT longitude, latitude FROM Checkins WHERE userid = ?', userid)
    locations = cursor.fetchall()
    longitudes = np.array([location[0] for location in locations])
    latitudes = np.array([location[1] for location in locations])

    # plot arrows
    plt.figure()
    plt.quiver(longitudes[:-1], latitudes[:-1], longitudes[1:]-longitudes[:-1], latitudes[1:]-latitudes[:-1],
               scale_units='xy', angles='xy', scale=1, width=0.001, headwidth=10, headlength=5)
    plt.show()

    # Close the connection to database.
    cursor.close()
    conn.close()


def plot_checkin_frequency_3dhistogram(region_range, delta):
    # Connect to sqlite database. Database file should be placed under current directory.
    conn = sq.connect('checkins.db')
    cursor = conn.cursor()

    # Get total number of checkins.
    cursor.execute('SELECT longitude, latitude FROM Checkins WHERE '
                   'longitude BETWEEN ' + str(region_range[0][0]) + ' AND ' + str(region_range[0][1]) + ' AND '
                   'latitude BETWEEN ' + str(region_range[1][0]) + ' AND ' + str(region_range[1][1]))
    locations = cursor.fetchall()
    lons = np.array([location[0] for location in locations])
    lats = np.array([location[1] for location in locations])

    bin_num = int(max((region_range[0][1]-region_range[0][0])/delta[0],
                      (region_range[1][1]-region_range[1][0])/delta[1]))
    print(bin_num)
    heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=bin_num)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

    # Close the connection to database.
    cursor.close()
    conn.close()


# if choose_dataset('Gowalla'):
if choose_dataset('Brightkite'):
    # reformat_checkins('checkins.txt', 'checkins_valid_reordered.txt')
    # reformat_edges('edges_preview.txt', 'edges_preview_in_sparse_matrix.dat')
    # split_checkins_by_user('checkins_valid_reordered.txt')
    # location = load_chekins('LOCATION')
    # uservalidity, startend, datetime, location, locatid = load_chekins()
    # edge = load_edges('edges_preview_in_sparse_matrix.dat')
    # plot_in_geoplotlib([[-74.042580, -73.691646], [40.538534, 40.912725]])
    # plot_locations_by_user()
    # checkin_num_distribution('userid', 'DESC', True)
    # checkin_num_distribution('date')
    # checkin_num_distribution('locatid', 'DESC', True)
    plot_locations_by_region([[-122.419415, -104.682105], [37.385773, 39.878664]])
    # plot_checkin_frequency_3dhistogram([[-74.264309, -74.042580], [40.538534, 40.912725]], [0.01, 0.01]),
    # plot_rectangles()
    # plot_trajectory_by_user(2)
