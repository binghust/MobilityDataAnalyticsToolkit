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
