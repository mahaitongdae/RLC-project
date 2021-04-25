from utils.plot_new.plot_utils.load_record import load_data
from utils.plot_new.plot_single_exp import single_plot
import numpy as np
import matplotlib.pyplot as plt
import os
def plot_others(data_all, exp_index):
    # if data_all['x_other'][0] != []:
    numpy_dict = {}
    keys = ['x_other', 'y_other', 'v_other', 'phi_other']
    y_lim_list = [100, 100, 5, 180]
    for key in keys:
        numpy_dict[key] = np.array(data_all[key])



    def single_plot_for_each_others(data, name, path):
        plt.figure()
        plt.plot(data_all['Time'], data)
        plt.title(name)
        plt.grid()
        # plt.ylim([-y_lim, y_lim])
        # plt.xlim([0,80])
        if not os.path.exists(path):
            os.mkdir(path)
        fig_path = path+name
        plt.savefig(fig_path)

    for i in range(3):
        for j in range(len(keys)):
            name = keys[j] + str(i) + '.jpg'
            proj_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            fig_path = proj_root_dir + '/record/' + exp_index + '/figure/data_fig/'
            single_plot_for_each_others(numpy_dict[keys[j]][:, i], name, fig_path)




if __name__ == '__main__':
    data_all, keys_for_data = load_data('left/case0_noise0_20210103_205805')
    plot_others(data_all, 'left/case0_noise0_20210103_205805')
