import numpy as np

dataset_dir = r'D:\Workspace\Datasets\Socail Network Dataset\Location-Based Social Network Dataset'

# get a new dataset by reordering the column loc_ids
def reorder_dataset(dataset_path_original, dataset_path_new):
    user_ids, loc_dates, loc_times, latitudes, longitudes, loc_ids_original = np.loadtxt(
        dataset_path_original,
        delimiter='\t',
        dtype=bytes,
        # ATTENTION: used to strip b'', maybe run out of memory
        # dtype= str,
        # converters={0:lambda x:x.decode(), 1:lambda x:x.decode(), 2:lambda x:x.decode(), 3:lambda x:x.decode(), 4:lambda x:x.decode()},
        unpack=True)

    loc_id_dict = dict(zip(set(loc_ids_original), range(0, len(loc_ids_original))))
    loc_ids_reordered = [loc_id_dict[loc_id] for loc_id in loc_ids_original]

    # ATTENTION: the first five columns in the new dataset contains b'', should be deleted manully
    np.savetxt(
        dataset_path_new,
        np.column_stack((user_ids, loc_dates, loc_times, latitudes, longitudes, loc_ids_reordered)),
        fmt='%s',
        delimiter='\t',
        newline='\r\n')

# brightkite_trajectory_original_path = dataset_dir + r'\SNAP Brightkite\Brightkite_totalCheckins.txt'
# brightkite_trajectory_new_path = dataset_dir + r'\SNAP Brightkite\Brightkite_totalCheckins_reordered.txt'
# reorder_dataset(brightkite_trajectory_original_path, brightkite_trajectory_new_path)

gowalla_trajectory_original_path = dataset_dir + r'\SNAP Gowalla\Gowalla_totalCheckins.txt'
gowalla_trajectory_new_path = dataset_dir + r'\SNAP Gowalla\Gowalla_totalCheckins_reordered.txt'
reorder_dataset(gowalla_trajectory_original_path, gowalla_trajectory_new_path)
