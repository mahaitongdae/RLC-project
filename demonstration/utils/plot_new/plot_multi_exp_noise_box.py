import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from utils.plot_new.plot_utils.load_record import load_data
from utils.plot_new.plot_utils.search_index import search_geq,search_leq,search_automode_index

def load_all_data(model_index, case):
    multi_exp_data = []
    for noise_factor in [0,1,2,3,4,5,6]:
        proj_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        short_exp_father_dir = 'noise'+str(noise_factor)
        full_exp_father_dir = proj_root_dir + '/utils/models/' + model_index + '/record/' + short_exp_father_dir
        exp_name = os.listdir(full_exp_father_dir)[0]
        exp_index = short_exp_father_dir + '/' + exp_name
        data, _ = load_data(model_index, exp_index)
        multi_exp_data.append(data)

        fig_dir_in_full_exp_father_dir = proj_root_dir + '/utils/models/' + model_index  + '/record' \
                                          + '/noise_figure'

        if not os.path.exists(fig_dir_in_full_exp_father_dir):
            os.mkdir(fig_dir_in_full_exp_father_dir)

    return multi_exp_data, fig_dir_in_full_exp_father_dir

def noise_box_plot(data, key, case, **kwargs):
    fontsize = 15
    df_list = []
    Noise=[0.0,1.0,2.0,3.0,4.0,5.0,6.0]
    for i in range(0, 7, 1):
        # min_index, max_index = search_automode_index(data[i])
        index_list = search_automode_index(data[i]['VehicleMode'])
        min_index, max_index = index_list[0], index_list[-1]
        print(i, min_index, max_index)
        PD = pd.DataFrame(dict(data=np.array(data[i][key][min_index:max_index]).squeeze(), Noise=i, ))
        df_list.append(PD)
    YawRate_dataframe = df_list[0].append(df_list[1:], ignore_index=True, )
    std_list = []
    for i in range(0, 7, 1):
        df4anoise = YawRate_dataframe[YawRate_dataframe['Noise'].isin([i])]
        std_list.append(np.std(df4anoise['data']))
    sns.set(style="darkgrid")
    f2 = plt.figure()
    ax2 = f2.add_axes([0.16, 0.14, 0.83, 0.85])
    title = 'case' + str(case)
    ax2.set_title(title)
    sns.boxplot(ax=ax2, x="Noise", y="data", data=YawRate_dataframe, palette="bright",
                order=np.arange(0, 7, 1))
    std_plot = ax2.plot(list(range(0, 7, )), std_list, color='indigo')
    if key == 'ego_vy':
        ax2.legend(handles=[std_plot[-1]], labels=['Standard variance'], loc='upper left', frameon=False)
    y_label = {'a_x': 'Acceleration [$\mathrm {m/s^2}$]',
               'SteerAngleAct': 'Steering Angle [$\circ$]',
               'tracking_delta_phi': 'Heading angle error',
               'tracking_delta_y': 'Position error [m]',
               'ego_vx': 'Speed [m/s]',
               'ego_vy': 'Lateral velocity [m/s]',
               'ego_r': 'Yaw rate [rad/s]',
               'ego_phi': 'Heading [$\circ$]',
               'tracking_delta_v': 'Velocity error [m/s]',
               'Time': 'Computing time [ms]',

               }
    ax2.set_xlabel('Noise level', fontsize=fontsize)
    ax2.set_ylabel(y_label[key], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    if 'path' in kwargs.keys():
        fig_name = kwargs['path'] + '/' + key + '.pdf'
        plt.savefig(fig_name)
    else:
        plt.show()


if __name__ == '__main__':
    # IMPORTANT:
    # support only one experiment in one noise directory, e.g., only exists 10_144620_real in noise0/.
    # pls delete redundant experiment directory.
    model_index = 'left/experiment-2021-01-16-10-34-37'
    # for case in [0,1,2]:
    case = 0
    for key in ['SteerAngleAct', 'a_x', 'ego_r', 'ego_vy']:
        multi_exp_data, fig_dir = load_all_data(model_index, case)
        noise_box_plot(multi_exp_data, key, case, path=fig_dir)
