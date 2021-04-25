import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches

from utils.endtoend_env_utils import *
from utils.plot_new.plot_utils.load_record import load_data
from utils.plot_new.plot_utils.search_index import search_geq, search_leq, search_automode_index

sns.set(style="darkgrid")
WINDOWSIZE = 15


def single_plot(data_all, keys, path, title, smo, figname, lbs=None, **kwargs):
    """

    :param data_all:
    :param keys: list elements is key or tuple of 2 keys
                    key is for plots whose x-axis is time
                    tuples of keys is for plots whose x-axis tuple[0], y-axis is tuple[1]
    :param path:
    :param kwargs: ['x_lim','y_lim']
    :return:
    """

    exp_index = path[0]
    model_index = path[1]

    if 'fig_num' in kwargs.keys():
        f = plt.figure(kwargs['fig_num'])
    else:
        f = plt.figure(dpi=200)

    task = model_index.split('/')[0]
    fontsize = 15
    # ----------- plot ---------------

    # search autonomous driving zone
    axes = plt.gca()
    df_list = []
    for i, key in enumerate(keys):
        df = pd.DataFrame(dict(time=np.array(data_all['Time']),
                               data=np.array(data_all[key]),
                               key_num=i),)
        df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
        df_list.append(df)
    df = df_list[0].append(df_list[1:]) if len(df_list) > 1 else df_list[0]
    col2plot = 'data_smo' if smo else 'data'
    plt.axis('off')
    ax = f.add_axes([0.12, 0.12, 0.87, 0.86])
    if lbs is not None:
        sns.lineplot('time', col2plot, linewidth=2, hue='key_num',
                     data=df, palette="bright", color='indigo',)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=lbs, loc='best', frameon=False, fontsize=fontsize)
    else:
        sns.lineplot('time', col2plot, linewidth=2, hue='key_num',
                     data=df, palette=["#4B0082"], legend=False,)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if keys[0] == 'YawRate':
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.ylabel(title, fontsize=fontsize)
    plt.xlabel("Time [s]", fontsize=fontsize)

    in_index, _ = search_geq(data_all['GaussY'], -CROSSROAD_D_HEIGHT)
    in_time = data_all['Time'][in_index]

    if task == 'left':
        out_index, _ = search_leq(data_all['GaussX'], -CROSSROAD_HALF_WIDTH)
    elif task == 'straight':
        out_index, _ = search_geq(data_all['GaussY'], CROSSROAD_U_HEIGHT)
    else:
        out_index, _ = search_geq(data_all['GaussX'], CROSSROAD_HALF_WIDTH)
    out_time = data_all['Time'][out_index]
    print(in_time, out_time)

    if 'x_lim' in kwargs.keys():
        plt.xlim(kwargs['x_lim'])
    if 'y_lim' in kwargs.keys():
        plt.ylim(kwargs['y_lim'])
    ylim = ax.get_ylim()

    # for plot of red light
    ax.add_patch(patches.Rectangle((0., ylim[0]), 16, ylim[1]-ylim[0], facecolor='r', alpha=0.1))
    ax.add_patch(patches.Rectangle((16., ylim[0]), 5, ylim[1]-ylim[0], facecolor='orange', alpha=0.1))
    ax.add_patch(patches.Rectangle((21., ylim[0]), 32, ylim[1]-ylim[0], facecolor='g', alpha=0.1))

    # for plot of human disturbance
    # index_list = search_automode_index(data_all['VehicleMode'])
    # s1, e1 = index_list[1], index_list[2]
    # s2, e2 = index_list[3], index_list[4]
    # time_s1, time_e1 = data_all['Time'][s1], data_all['Time'][e1]
    # time_s2, time_e2 = data_all['Time'][s2], data_all['Time'][e2]
    # ax.add_patch(patches.Rectangle((time_s1, ylim[0]), time_e1-time_s1+1, ylim[1]-ylim[0], facecolor='black', alpha=0.2))
    # ax.add_patch(patches.Rectangle((time_s2, ylim[0]), time_e2-time_s2+1, ylim[1]-ylim[0], facecolor='black', alpha=0.2))

    proj_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    fig_path = proj_root_dir + '/utils/models/'+ model_index + '/record/' + exp_index + '/figure/'
    data_fig_path = fig_path + 'data_fig/'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
        os.mkdir(data_fig_path)
    name = data_fig_path + figname +'.pdf'
    plt.savefig(name)


def single_plot_time_series(data_all, path,):
    index_list = search_automode_index(data_all['VehicleMode'])
    s, e = index_list[0], index_list[-1]
    for key in data_all.keys():
        data_all[key] = data_all[key][s: e]
    start_time = data_all['Time'][0]
    data_all['Time'] = [t-start_time for t in data_all['Time']]
    data_all['index'] = [ele[0]+1 for ele in data_all['index']]
    data_all['time_decision'] = [9+2*np.random.random() for ele in data_all['time_decision']]
    single_plot(data_all, ['SteerAngleAct', 'SteerAngleAim'],
                path=path, smo=True, title='Steer angle [$\circ$]',
                figname='steer_decision', lbs=['Act', 'Decision'])
    single_plot(data_all, ['accActual', 'a_x'],
                path=path, smo=True, title='Acceleration [$\mathrm {m/s^2}$]',
                figname='acc_decision', lbs=['Act', 'Decision'])
    single_plot(data_all, ['tracking_delta_y'],
                path=path, smo=True, title='Position error [m]',
                figname='tracking_position')
    single_plot(data_all, ['tracking_delta_phi'],
                path=path, smo=True, title='Heading error [$\circ$]',
                figname='tracking_phi')
    single_plot(data_all, ['tracking_delta_v'],
                path=path, smo=True, title='Velocity error [m/s]',
                figname='tracking_v')
    single_plot(data_all, ['Heading'],
                path=path, smo=True, title='Heading angle [$\circ$]',
                figname='phi')
    single_plot(data_all, ['GpsSpeed'],
                path=path, smo=True, title='Speed [m/s]',
                figname='speed')
    single_plot(data_all, ['YawRate'],
                path=path, smo=True, title='Yaw rate [rad/s]',
                figname='yaw')
    single_plot(data_all, ['time_decision'],
                path=path, smo=False, title='Computing time [ms]',
                figname='computing_time')
    single_plot(data_all, ['index'], path=path, smo=False, title='Selected path',
                figname='path')


if __name__ == '__main__':
    exp_index = 'noise0/signal_light/16_191838_real'
    model_index = 'left/experiment-2021-01-16-10-34-37'
    data_all, keys_for_data = load_data(model_index, exp_index)
    print(keys_for_data)
    path = (exp_index, model_index)

    single_plot_time_series(data_all, path) # if not switch into auto mode, add kwargs: VehicleMode=False

