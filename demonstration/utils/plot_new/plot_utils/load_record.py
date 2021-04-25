import re
import os
import numpy as np
import sys

list_data = ['State_other','Path']
# 获取文件的内容
def get_contents(path):
    with open(path,'r', encoding='UTF-8') as file_object:
        contents = file_object.readlines()
    with open(path, 'r', encoding='UTF-8') as file_object:
        str_contents = file_object.read()
    return contents, str_contents

def load_data(model_dir, record_dir):
    """
    :param record_dir:
    :return:
    """
    # get project root directory
    proj_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    path = proj_root_dir + '/utils/models/'+ model_dir + '/record/' + record_dir + '/record.txt'
    # try:
    contents, str_contents = get_contents(path)
    # except:
    #     print('FILE NOT FOUND! Check file path, e.g. case, noise, task')
    #     sys.exit(1)
    data_all_dict = {}
    keys_for_data = {'Decision':[], 'State_ego':[],'State_other':[], 'Obs_dict':[], 'Path':[], 'Time':[]} # 'Obs_dict':[],
    for row in contents[0:len(keys_for_data.keys()) + 2]:
        for keys_class in keys_for_data.keys(): #
            if keys_class in row:
                if keys_class in list_data:
                    for i, d in enumerate(row.split('|')[:-1]):
                        data_key = d.split(':')[0]
                        data_key = data_key.split(' ')[1]
                        keys_for_data[keys_class].append(data_key)
                        data_all_dict[data_key] = []
                else:
                    for i, d in enumerate(row.split(',')[:-1]): # -1 is \n
                        data_key = d.split(':')[0]
                        data_key = data_key.split(' ')[1]
                        keys_for_data[keys_class].append(data_key)
                        data_all_dict[data_key] = []
    keys_for_data['State_ego'].append('accActual')
    for row in contents:
        for keys_class in keys_for_data.keys(): #
            if keys_class in row:
                if keys_class in list_data:
                    for i, d in enumerate(row.split('|')[:-1]):
                        data = d.split(':')[1]
                        data_all_dict[keys_for_data[keys_class][i]].append(eval(data))
                else:
                    for i, d in enumerate(row.split(',')[:-1]):
                        data = d.split(':')[1]
                        data_all_dict[keys_for_data[keys_class][i]].append(float(data))

    data_all_dict['accActual'] = []
    for i in range(len(data_all_dict['GpsSpeed'])-1):
        # print(data_all_dict['GpsSpeed'])
        data_all_dict['accActual'].append((float(data_all_dict['GpsSpeed'][i+1]) - float(data_all_dict['GpsSpeed'][i])) / (data_all_dict['Time'][i+1]-data_all_dict['Time'][i]))
    data_all_dict['accActual'].append(data_all_dict['accActual'][-1])


    return data_all_dict, keys_for_data



def atest_load_txt():
    data_dict, keys_for_data = load_data('left/experiment-2021-01-04-01-15-35','left_case0_20210103_121512')
    a = 1


if __name__ == '__main__':
    atest_load_txt()
