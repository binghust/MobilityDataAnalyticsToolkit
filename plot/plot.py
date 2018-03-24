import sqlite3 as sq

import matplotlib.pyplot as plt


class KAnonymousLocationsPlot:
    def __init__(self):
        self.K = 6
        self.conn = sq.connect('checkins.db')
        self.cursor = self.conn.cursor()
        self.ano_locs = None
        self.tablename = None

    def print_anonymous_locations(self):
        x_actuals = []
        y_actuals = []
        x_dummies = []
        y_dummies = []
        plt.axis([-1, 30, -1, 30])
        plt.ion()
        plt.show()

        length = len(self.ano_locs)
        i = -1
        while i < length:
            command = 'n'
            if command == 'n' and i < length:
                i += 1
                if len(x_actuals) > 0:
                    plt.scatter(x_dummies, y_dummies, c='g', marker='x')
                    plt.scatter(x_actuals, y_actuals, c='r', marker='x')
                for j in range(1, self.K):
                    plt.scatter(self.ano_locs[i][j][0], self.ano_locs[i][j][1], c='b', marker='x')
                plt.scatter(self.ano_locs[i][0][0], self.ano_locs[i][0][1], c='y', marker='x')
                x_actuals.append(self.ano_locs[i][0][0])
                y_actuals.append(self.ano_locs[i][0][1])
                for j in range(1, self.K):
                    x_dummies.append(self.ano_locs[i][j][0])
                    y_dummies.append(self.ano_locs[i][j][1])
            elif command == 'p' and i > -1:
                i -= 1
                x_actuals.pop()
                y_actuals.pop()
                for j in range(1, self.K):
                    x_dummies.pop()
                    y_dummies.pop()
                if len(x_actuals) > 0:
                    plt.scatter(x_dummies, y_dummies, c='g', marker='x')
                    plt.scatter(x_actuals, y_actuals, c='r', marker='x')
                for j in range(1, self.K):
                    plt.scatter(self.ano_locs[i][j][0], self.ano_locs[i][j][1], c='b', marker='x')
                plt.scatter(self.ano_locs[i][0][0], self.ano_locs[i][0][1], c='y', marker='x')
            else:
                break
            plt.draw()
            plt.pause(1)

    def get_anonymous_locations(self, userid):
        sql_fetchall = ''.join(['SELECT',
                                ''.join([''.join([' gridlon', str(i), ', gridlat', str(i), ',']) for i in
                                         range(0, self.K - 1)]),
                                ' gridlon', str(self.K - 1), ', gridlat', str(self.K - 1),
                                ' FROM ',
                                self.tablename,
                                ' WHERE userid = ',
                                str(userid),
                                ' ORDER BY locdatetime ASC'])
        self.cursor.execute(sql_fetchall)
        results = self.cursor.fetchall()
        anonymous_locs = []
        for record in results:
            anonymous_loc = []
            for i in range(0, 2 * self.K, 2):
                anonymous_loc.append((record[i], record[i + 1]))
            anonymous_locs.append(anonymous_loc)
        return anonymous_locs



#
#
# def plot_checkin_num_distribution_by_user():
#     # Connect to sqlite database. Database file should be placed under current directory.
#     conn = sq.connect('checkins.db')
#     cursor = conn.cursor()
#
#     # Get checkin count.
#     cursor.execute(''.join(['SELECT count(*)',
#                             ' FROM ', Dataset.tablename]))
#     result = cursor.fetchall()
#     checkin_num = result[0][0]
#
#     # Get checkin count of each user.
#     cursor.execute(''.join(['SELECT userid, count(*)',
#                             ' FROM ', Dataset.tablename,
#                             ' GROUP BY userid',
#                             ' ORDER BY count(*) DESC']))
#     result = cursor.fetchall()
#     userids = np.array([userid for userid, checkin_num in result])
#     user_num = userids.size
#     checkin_nums_by_userid = np.array([checkin_num for userid, checkin_num in result])
#
#     # Calulate cumulative distribution of checkin count.
#     checkin_cnt = 0
#     cumulative_distribution = np.empty(user_num, dtype=np.float16)
#     for i in range(0, user_num):
#         checkin_cnt += checkin_nums_by_userid[i]
#         cumulative_distribution[i] = checkin_cnt / checkin_num * 100
#
#     # Plot distribution and cumulative distribution of checkin number in the same figure.
#     xaxis = np.linspace(1, user_num, user_num, dtype=np.uint32)
#     fig = plt.figure(0)
#     ax1 = fig.add_subplot(111)
#     ax1.set_title('Distribution and Cumulative Distribution of Checkin Count')
#     ax1.set_xlabel('number of user')
#     ax1.set_ylabel('percent of checkin count (%)', color='b')
#     ax1.tick_params('y', colors='b')
#     ax1.plot(xaxis, cumulative_distribution, color='b')
#     ax2 = plt.twinx()
#     ax2.set_ylabel('checkin count', color='k')
#     ax2.tick_params('y', colors='k')
#     ax2.scatter(xaxis, checkin_nums_by_userid, color='k')
#     fig.show()
#
#     # Plot distribution and cumulative distribution of checkin count in different figures.
#     xaxis = np.linspace(1, user_num, user_num, dtype=np.uint32)
#     fig_cumulative_distribution = plt.figure(1)
#     ax_cumulative_distribution = fig_cumulative_distribution.add_subplot(111)
#     ax_cumulative_distribution.set_title('Cumulative Distribution of Checkin Count')
#     ax_cumulative_distribution.set_xlabel('user count')
#     ax_cumulative_distribution.set_ylabel('percent of checkin count (%)', color='b')
#     ax_cumulative_distribution.plot(xaxis, cumulative_distribution, color='b')
#     fig_cumulative_distribution.show()
#     fig_distribution = plt.figure(2)
#     ax_distribution = fig_distribution.add_subplot(111)
#     ax_distribution.set_title('Distribution of Checkin Count')
#     ax_distribution.set_xlabel('user count')
#     ax_distribution.set_ylabel('checkin count', color='k')
#     ax_distribution.scatter(xaxis, checkin_nums_by_userid, color='k')
#     fig_distribution.show()
#
#     plt.show()
#
#     # Close connection to database.
#     cursor.close()
#     conn.close()
#
#
# def plot_in_geoplotlib(region_range):
#     """
#     Plot a region on map using geoplotlib
#     :param region_range: 2x2 list, [0][0]: minlongitude, [0][1]: maxlongitude, [1][0]: minlatitude, [1][1]: maxlatitude
#     :return: None
#     """
#     db_access geoplotlib as gp
#
#     # Connect to sqlite database. Database file should be placed in current directory.
#     conn = sq.connect('checkins.db')
#     cursor = conn.cursor()
#
#     # Get locations within region_range.
#     cursor.execute(''.join(['SELECT lon, lat',
#                             ' FROM ', Dataset.tablename,
#                             ' WHERE lon BETWEEN ? AND ? AND latitude BETWEEN ? AND ?']),
#                    (region_range[0][0], region_range[0][1], region_range[1][0], region_range[1][1]))
#     locations = cursor.fetchall()
#     # Use numpy.ndarray instead of list, otherwise it failed.
#     longitudes = np.array([location[0] for location in locations])
#     latitudes = np.array([location[1] for location in locations])
#
#     gp.hist(gp.utils.DataAccessObject({'lon': longitudes, 'lat': latitudes}), colorscale='sqrt', binsize=8)
#     gp.show()
#
#     # Close the connection to database.
#     cursor.close()
#     conn.close()
#
#
# def plot_locations_by_user():
#     # Connect to sqlite database. Database file should be placed under current directory.
#     conn = sq.connect('checkins.db')
#     cursor = conn.cursor()
#
#     # Get user_ids of all of users.
#     cursor.execute(''.join(['SELECT DISTINCT userid',
#                             ' FROM ', Dataset.tablename]))
#     user_ids = cursor.fetchall()
#
#     # Generate a color for each user's plot
#     colormap = plt.get_cmap('gnuplot')
#     colors = [colormap(i) for i in np.linspace(0, 1, len(user_ids))]
#
#     for user_id, color in zip(user_ids, colors):
#         # Get locations of each user.
#         cursor.execute(''.join(['SELECT lon, lat',
#                                 ' FROM ', Dataset.tablename,
#                                 ' WHERE userid = ?']), (user_id[0],))
#         locations = np.array(cursor.fetchall())
#         # Plot locations of each user by regarding longitude as X-axis, latitude as Y-axis.
#         # Plotting costs much more time than querying database does.
#         plt.scatter(locations[:, 0], locations[:, 1], c=color, marker='o')
#     plt.show()
#
#     # Close the connection to database.
#     cursor.close()
#     conn.close()
#
#
# def plot_locations_by_region(region_range):
#     # Connect to sqlite database. Database file should be placed under current directory.
#     conn = sq.connect('checkins.db')
#     cursor = conn.cursor()
#
#     # Get total number of checkins.
#     cursor.execute('SELECT lon, lat FROM Checkins_Preview '
#                    'WHERE longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?',
#                    (region_range[0][0], region_range[0][1], region_range[1][0], region_range[1][1]))
#     locations = cursor.fetchall()
#     longitudes = np.array([location[0] for location in locations])
#     latitudes = np.array([location[1] for location in locations])
#     plt.scatter(longitudes, latitudes, c='b', marker='o')
#     plt.show()
#
#     # Close the connection to database.
#     cursor.close()
#     conn.close()
#
#
# def generate_rectangle():
#     from matplotlib.patches db_access Rectangle
#
#     # Connect to sqlite database. Database file should be placed under current directory.
#     conn = sq.connect('checkins.db')
#     cursor = conn.cursor()
#
#     # Get user_ids of all of users.
#     cursor.execute('SELECT latitude, longitude FROM Checkins_Preview')
#     locations = cursor.fetchall()
#     latitudes = np.array([location[0] for location in locations])
#     longitudes = np.array([location[1] for location in locations])
#
#     point_num_in_rec = 4
#     rectangles = []
#     for i in range(0, len(locations), point_num_in_rec):
#         top = max(latitudes[i:i + point_num_in_rec])
#         bottom = min(latitudes[i:i + point_num_in_rec])
#         left = min(longitudes[i:i + point_num_in_rec])
#         right = max(longitudes[i:i + point_num_in_rec])
#         rectangle = Rectangle((left, bottom), right - left, top - bottom, fill=False)
#         rectangles.append(rectangle)
#
#     # Close the connection to database.
#     cursor.close()
#     conn.close()
#
#     return rectangles
#
#
# def plot_rectangles():
#     fig = plt.figure(1)
#     axes = fig.add_subplot(111)
#     rectangles = generate_rectangle()
#     for rectangle in rectangles:
#         axes.add_patch(rectangle)
#     plt.plot(-122.419415, 37.774929)
#     plt.show()
#
#
# def plot_trajectory_by_user(userid):
#     # Connect to sqlite database. Database file should be placed under current directory.
#     conn = sq.connect('checkins.db')
#     cursor = conn.cursor()
#
#     # Get total number of checkins.
#     cursor.execute('SELECT longitude, latitude FROM Checkins WHERE userid = ?', userid)
#     locations = cursor.fetchall()
#     longitudes = np.array([location[0] for location in locations])
#     latitudes = np.array([location[1] for location in locations])
#
#     # plot arrows
#     plt.figure()
#     plt.quiver(longitudes[:-1], latitudes[:-1], longitudes[1:] - longitudes[:-1], latitudes[1:] - latitudes[:-1],
#                scale_units='xy', angles='xy', scale=1, width=0.001, headwidth=10, headlength=5)
#     plt.show()
#
#     # Close the connection to database.
#     cursor.close()
#     conn.close()
#
#
# def plot_checkin_frequency_3dhistogram(region_range, delta):
#     """
#     Args:
#         region_range: 2x2 list, [0][0]: minlongitude, [0][1]maxlongitude, [1][0]: minlatitude, [1][1]: maxlatitude.
#         delta: 1x2 list, [0]: deltalongitude, [1]: deltalatitude.
#     Return:
#         None
#     """
#     # Connect to sqlite database. Database file should be placed under current directory.
#     conn = sq.connect('checkins.db')
#     cursor = conn.cursor()
#
#     # Get total number of checkins.
#     cursor.execute('SELECT longitude, latitude FROM Checkins'
#                    ' WHERE longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?',
#                    (region_range[0][0], region_range[0][1], region_range[1][0], region_range[1][1]))
#     locations = cursor.fetchall()
#     lons = np.array([location[0] for location in locations])
#     lats = np.array([location[1] for location in locations])
#
#     bin_num = int(max((region_range[0][1] - region_range[0][0]) / delta[0],
#                       (region_range[1][1] - region_range[1][0]) / delta[1]))
#     print(bin_num)
#     heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=bin_num)
#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#
#     plt.clf()
#     plt.imshow(heatmap.T, extent=extent, origin='lower')
#     plt.show()
#
#     # Close the connection to database.
#     cursor.close()
#     conn.close()
#
#
# if choose_dataset('Gowalla'):
#     # if choose_dataset('Brightkite'):
#     # edge = load_edges('edges_preview_in_sparse_matrix.dat')
#     plot_in_geoplotlib([[-122.521368, -122.356684], [37.706357, 37.817344]])  # Brooklyn of New York City
#     # plot_locations_by_user()
#     # plot_checkin_num_distribution_by_user()
#     # plot_checkin_num_distribution('date')
#     # plot_checkin_num_distribution('locatid'
#     # plot_locations_by_region([[-122.419415, -104.682105], [37.385773, 39.878664]])
#     # plot_checkin_frequency_3dhistogram([[-74.264309, -74.042580], [40.538534, 40.912725]], [0.01, 0.01]),
#     # plot_rectangles()
#     # plot_trajectory_by_user(2)
