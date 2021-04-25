#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/27
# @Author  : Yangang Ren (Tsinghua Univ.)
# @FileName: Online application of trained network.py
# =====================================

from __future__ import print_function

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from multiprocessing import Process

from controller import Controller
from plot_online import Plot
from subscriber_can import SubscriberCan
from subscriber_gps import SubscriberGps
from traffic_sumo import Traffic
from render_online import Render

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# gps,can->traffic->controller->plot


def controller_agent(shared_list, receive_index, lock, args):
    publisher_ = Controller(shared_list, receive_index, lock, args)
    time.sleep(10)
    publisher_.run()


def subscriber_can_agent(shared_list, receive_index, lock):
    subscriber_ = SubscriberCan(shared_list, receive_index, lock)
    subscriber_.run()


def subscriber_gps_agent(shared_list, receive_index, lock):
    subscriber_ = SubscriberGps(shared_list, receive_index, lock)
    subscriber_.run()


def traffic(shared_list, lock, step_length, mode, model_only_test, task):
    subscriber_ = Traffic(shared_list, lock, step_length, mode, model_only_test, task)
    subscriber_.run()


def plot_agent(shared_list, lock, args):
    if args.visualization == 'plot':
        plot_ = Plot(shared_list, lock, args)
        time.sleep(16)
        plot_.run()
    else:
        render_ = Render(shared_list, lock, args)
        time.sleep(16)
        render_.run()


def built_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='left')
    parser.add_argument('--if_save', type=bool, default=True)
    task = parser.parse_args().task
    if task == 'left':
        parser.add_argument('--load_dir', type=str, default='./utils/models/{}/experiment-2021-01-16-10-34-37'.format(task))
        parser.add_argument('--load_ite', type=str, default=100000)
    elif task == 'right':
        parser.add_argument('--load_dir', type=str, default='./utils/models/{}/experiment-2021-01-16-09-53-14'.format(task))
        parser.add_argument('--load_ite', type=str, default=55000)
    elif task == 'straight':
        parser.add_argument('--load_dir', type=str, default='./utils/models/{}/experiment-2021-01-16-12-10-42'.format(task))
        parser.add_argument('--load_ite', type=str, default=80000)
    parser.add_argument('--visualization', type=str, default='render')  # plot or render

    parser.add_argument('--noise_factor', type=float, default=6)
    parser.add_argument('--model_only_test', type=bool, default=False)
    parser.add_argument('--traffic_step_length', type=float, default=100.)
    parser.add_argument('--traffic_mode', type=str, default='training')
    parser.add_argument('--clipped_v', type=float, default=5., help='m/s')
    parser.add_argument('--true_ss', type=str, default='con_v')  # None, pred, con_v
    parser.add_argument('--ss_con_v', type=float, default=5.)

    parser.add_argument('--backup', type=str, default='test')

    load_dir = parser.parse_args().load_dir
    model_only_test = parser.parse_args().model_only_test
    flag = 'model' if model_only_test else 'real'
    noise = int(parser.parse_args().noise_factor)
    result_dir = load_dir + '/record/noise{noise}/{time}_{flag}'.format(
        noise=noise,
        time=datetime.now().strftime("%d_%H%M%S"),
        flag=flag)
    parser.add_argument('--result_dir', type=str, default=result_dir)
    return parser.parse_args()


def main():
    args = built_parser()
    os.makedirs(args.result_dir)
    with open(args.result_dir + '/config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    shared_list = mp.Manager().list([0.] * 15)
    # [state_gps, time_gps, state_can, time_can, state_other, time_radar,
    #  step, runtime, decision, state_ego, obs_vec, traj_value, ref_index, v_light]

    # state_gps['GaussX'] = 0   # intersection coordinate [m]
    # state_gps['GaussY'] = 0   # intersection coordinate [m]
    # state_gps['Heading'] = 0  # intersection coordinate [deg]
    # state_gps['GpsSpeed'] = 0          # [m/s]
    # state_gps['NorthVelocity'] = 0     # [m/s]
    # state_gps['EastVelocity'] = 0      # [m/s]
    # state_gps['YawRate'] = 0           # [rad/s]
    # state_gps['LongitudinalAcc'] = 0   # [m/s^2]
    # state_gps['LateralAcc'] = 0        # [m/s^2]
    # state_gps['Longitude'] = 0
    # state_gps['Latitude'] = 0

    # state_can['VehicleSPeedAct'] = 0  # [m/s]
    # state_can['SteerAngleAct'] = 0    # [m/s]
    # state_can['AutoGear'] = 0
    # state_can['VehicleMode'] = 0      # 0: manual driving; 1: autonomous driving
    # state_can['Throttle'] = 0
    # state_can['BrkOn'] = 0

    # state_other: dict(x_other=[],  # intersection coordination
    #                   y_other=[],  # intersection coordination
    #                   v_other=[],
    #                   phi_other=[],  # intersection coordination
    #                   v_light=0)

    # decision: {'Deceleration': decel,  # [m/s^2]
    #             'Torque': torque,  # [N*m]
    #             'Dec_flag': dec_flag,
    #             'Tor_flag': tor_flag,
    #             'SteerAngleAim': steer_wheel_deg,  # [deg]
    #             'front_wheel_deg': front_wheel_deg,
    #             'a_x': a_x})  # [m/s^2]

    receive_index = mp.Value('d', 0.0)
    lock = mp.Lock()
    procs = [Process(target=subscriber_gps_agent, args=(shared_list, receive_index, lock)),
             Process(target=subscriber_can_agent, args=(shared_list, receive_index, lock)),
             Process(target=traffic, args=(shared_list, lock, args.traffic_step_length, args.traffic_mode,
                                           args.model_only_test, args.task)),
             Process(target=controller_agent, args=(shared_list, receive_index, lock, args)),
             Process(target=plot_agent, args=(shared_list, lock, args)),]

    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
