# encoding=utf-8
import os
import preprocessing as p

# os.chdir('D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Brightkite')
os.chdir('D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Gowalla')

# # dump data in csv files into database.
# p.Preprocessing.reformat_checkins('checkins.txt', 'checkins_valid_reordered.txt')
# p.Preprocessing.split_checkins_by_user('checkins_valid_reordered.txt')
# uservalidity, startend, datetime, location, locatid = p.Preprocessing.load_chekins()
# edge = p.Preprocessing.load_edges('edges_preview_in_sparse_matrix.dat')

preprocessing = p.Preprocessing('checkins.db', 'Checkins', 'Edges')

# checkins preprocessing.
target_checkins = preprocessing.start()
target_checkins = preprocessing.filter_by_region(target_checkins, p.map_)
target_checkins = preprocessing.filter_by_datetime(target_checkins, p.datetime_range)
target_checkins = preprocessing.subsample(target_checkins, p.subsample_interval)
target_checkins = preprocessing.filter_by_tracelength(target_checkins, p.min_tracelength)
target_checkins = preprocessing.discretization(target_checkins, p.grid_map)
preprocessing.create_checkins_index(target_checkins)

# edges preprocessing
target_edges = preprocessing.source_edges + target_checkins.replace(preprocessing.source_checkins, '')
preprocessing.filter_edges(preprocessing.source_edges, target_edges, target_checkins)

# Clean temp tables, compact the database file and close connection to database.
preprocessing.stop(clean=True, compact=True)
