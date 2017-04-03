# Reorder an array
def reorder_array(source_array):
    element_dict = dict(zip(set(source_array), range(0, len(source_array))))
    target_array = [element_dict[element] for element in source_array]
    return target_array


# Get a new dataset by reordering the column loc_ids
# implemented in numpy.loadtxt, the first 5 columns in the new dataset contains b'', need to be deleted manually
# could also be implemented in pandas.read_csv
def reformat_dataset(source_path, target_path):
    #  Open file
    with open(source_path, 'r') as source_fid:
        # \r, \n and \r\n will be substituted by \n while reading file if the parameter newline is not specified
        # Load the original dataset to a row_num x 6 list
        data = [row.rstrip('\n').split('\t') for row in source_fid]

        # Delete rows including invalid coordinate such as lat/lon==0, lat<-90/>90, or lon<-180/>180
        invalid_row_index = []
        for row_index in range(len(data)-1, -1, -1):
            latitude = float(data[row_index][3])
            longitude = float(data[row_index][4])
            if (latitude == 0.0 or latitude <= -90.0 or latitude >= 90.0) and \
                    (longitude == 0.0 or longitude < -180.0 or longitude > 180.0):
                invalid_row_index.append(row_index + 1)
                del data[row_index]
        print('Invalid row(s):', invalid_row_index)

        # Reverse six columns to keep chronological order
        data = data[::-1]

        # Reorder the 5th column loc_id as a int list loc_ids_reordered
        loc_ids_raw = [row[5] for row in data]
        loc_ids_reordered = reorder_array(loc_ids_raw)

        # Write to a new dataset
        delimiter = '\t'
        with open(target_path, 'w') as target_fid:
            for i in range(0, len(data)):
                # \n will be substituted by \r, \n or \r\n  while writing file according to your operation system
                # If the parameter newline is not specified
                row = delimiter.join(data[i][0:5]) + delimiter + str(loc_ids_reordered[i]) + '\n'
                # If ended in '\r\n', it will be substituted by '\r\r\n'
                target_fid.write(row)


def split_by_user(source_path):
    #  Open file
    with open(source_path, 'r') as source_fid:
        data_by_user = []
        data_of_one_user = ''
        user_id_previous = ''
        for row in source_fid:
            user_id = row.rstrip('\n').split('\t')[0]
            if user_id != user_id_previous:
                data_by_user.append(data_of_one_user)
                data_of_one_user = row
            else:
                data_of_one_user += row
            user_id_previous = user_id
        # Delete the first element in data_by_user since it is ''.
        del data_by_user[0]
        # Append data of the last user to data_by_user
        data_by_user.append(data_of_one_user)


# reformat_dataset(
#     r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Gowalla\checkins_preview.txt',
#     r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Gowalla\checkins_preview_valid_reordered.txt')

# reformat_dataset(
#     r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Brightkite\checkins_preview.txt',
#     r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Brightkite\checkins_preview_valid_reordered.txt')

# reformat_dataset(
#     r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Gowalla\checkins.txt',
#     r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Gowalla\checkins_valid_reordered.txt')

# reformat_dataset(
#     r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Brightkite\checkins.txt',
#     r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Brightkite\checkins_valid_reordered.txt')

split_by_user(
    r'D:\Workspace\Datasets\Location-Based Social Network\SNAP Brightkite\checkins_preview_valid_reordered.txt')
