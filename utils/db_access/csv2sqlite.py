# encoding=utf-8
from datetime import datetime
from pickle import dump, load

import numpy as np
from scipy.io import mmread
from scipy.sparse import dok_matrix


def reorder_array(source_array):
    """
    Reorder an array by unique element
    :param source_array: 1xn list
    :return:
    """
    element_dict = dict(zip(set(source_array), range(0, len(source_array))))
    target_array = [element_dict[element] for element in source_array]
    return target_array


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
             (
                 'user_validity.dat', 'startends.dat', 'datetimes.dat', 'locations.dat',
                 'locat_ids.dat'))


def save(variables, filenames):
    """
    Serilize variable(s) to current directory according to given filename(s).
    :param variables: 1xn list, variable(s) that need to be saved.
    :param filenames: 1xn list, filename(s) with respective to each variable.
    :return: boolean
    """
    var_num = len(variables)
    if var_num == 0 or var_num != len(filenames):
        return False
    for i in range(0, var_num):
        with open(filenames[i], 'wb') as fid:
            dump(variables[i], fid)
    return True


def load_chekins(attribute_name=None):
    # load serilized variables(checkins) from disk to memory.
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


def load_edges(source_name):
    # load serilized variables(edges) from disk to memory.
    with open(source_name, 'rb') as fid:
        # Scipy.io.load returns a coo_matrix instead of a dok_matrix.
        return dok_matrix(mmread(fid), dtype=np.int8)

if __name__ == '__main__':
    pass
