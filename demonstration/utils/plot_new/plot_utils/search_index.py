import numpy as np
from utils.plot_new.plot_utils.load_record import load_data
def search_geq(data, threshold):
    """

    :param data: list or numpy 1d
    :param threshold:
    :return:
    """
    if isinstance(data, list):
        data = np.array(data)

    min = np.min(data)
    max = np.max(data)
    if threshold < min or threshold > max:
        print('threshold not in range of values')
        print('max is {:.2f}, min is {:.2f}'.format(float(max), float(min)))
        raise ValueError

    for i in range(data.shape[0]):
        if data[i] >= threshold:
            return i, data[i]

def search_leq(data, threshold):
    """

    :param data: list or numpy 1d
    :param threshold:
    :return:
    """
    if isinstance(data, list):
        data = np.array(data)

    min = np.min(data)
    max = np.max(data)
    if threshold < min or threshold > max:
        print('threshold not in range of values')
        print('max is {:.2f}, min is {:.2f}'.format(float(max), float(min)))
        raise ValueError

    for i in range(data.shape[0]):
        if data[i] <= threshold:
            return i, data[i]

def search_automode_index(data):
    index_list = []
    for i in range(len(data)-1):
        if data[i] != data[i+1]:
            index_list.append(i)

    return index_list
    # min_index = data['VehicleMode'].index(1.0)
    # max_index = min_index + data['VehicleMode'].count(1.0)
    # if max_index == len(data['VehicleMode']):
    #     max_index = -1
    #
    # return min_index, max_index

# def search_automode_time(data):
#
#     return map(lambda x: data["Time"][x])

if __name__ == '__main__':
    data_all, keys_for_data = load_data('left_case0_20210102_170343')
    ego_y = data_all['GaussY']
    i, x = search_geq(ego_y, 90)
    print(i, x)
