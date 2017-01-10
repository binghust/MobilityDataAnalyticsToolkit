import numpy as np


# Reorder an array
def reorder_array(source_array):
    element_dict = dict(zip(set(source_array), range(0, len(source_array))))
    target_array = np.array([element_dict[element] for element in source_array])
    return target_array


# Get a new dataset by reordering the column loc_ids
# implemented in numpy.loadtxt, the first 5 columns in the new dataset contains b'', need to be deleted manually
# could also be implemented in pandas.read_csv
def reformat_dataset(source_path, target_path):
    #  load six columns of the original dataset to a N x 6 ndarray
    six_columns = np.loadtxt(
        source_path,
        delimiter='\t',
        dtype=bytes)

    # delete rows including invalid coordinate (0.0, 0.0) or (0, 0)
    invalid_row_index = []
    for i, latitude in enumerate(six_columns[:, 3]):
        if (latitude == b'0.0' or latitude == b'0') and (six_columns[i, 4] == b'0.0' or six_columns[i, 4] == b'0'):
            invalid_row_index.append(i)
    six_columns = np.delete(six_columns, invalid_row_index, axis=0)  # axis=0: delete by row, axis=1: delete by column

    # reverse six columns to keep chronological order
    six_columns = six_columns[::-1]

    # reorder the column loc_id as a int list loc_ids_reordered
    loc_ids_reordered = reorder_array(six_columns[:, 5])

    # decode first five columns
    five_columns = [
        [x.decode('utf-8') for x in six_columns[row_no, [0, 1, 2, 3, 4]]] for row_no in range(0, len(six_columns))]

    np.savetxt(
        target_path,
        np.column_stack((five_columns, loc_ids_reordered)),
        fmt='%s',
        delimiter='\t',
        newline='\r\n')


# reformat_dataset(
#     r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Gowalla\checkins.txt',
#     r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Gowalla\checkins_valid_reordered.txt')

reformat_dataset(
    r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Brightkite\checkins.txt',
    r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Brightkite\checkins_valid_reordered.txt')
