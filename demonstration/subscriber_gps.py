#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/28
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: subscriber_gps.py
# =====================================

import json
import math
import time
import numpy as np
from collections import OrderedDict

import zmq

from utils.coordi_convert import convert_gps_coordi_to_intersection_coordi


class SubscriberGps():
    def __init__(self, shared_list, receive_index, lock):
        self.shared_list = shared_list
        self.receive_index_shared = receive_index
        self.receive_index = 0
        self.time_start_gps = 0.
        self.lock = lock
        self.x_bias = 1.4

        # communication
        context = zmq.Context()
        self.socket_gps = context.socket(zmq.SUB)
        self.socket_gps.connect("tcp://127.0.0.1:6666")                # 上车
        self.socket_gps.setsockopt(zmq.SUBSCRIBE, "".encode('utf-8'))  # 接收所有消息

    def rotate_and_move(self, x_rear_axle, y_rear_axle, heading):
        # RTK传来的自车位置不是在车辆正中心（车辆后轴）,需要换算到车辆的正中心
        temp_X, temp_Y, heading = convert_gps_coordi_to_intersection_coordi(x_rear_axle, y_rear_axle, heading)
        x_modified = temp_X + self.x_bias * math.cos(heading * math.pi / 180)
        y_modified = temp_Y + self.x_bias * math.sin(heading * math.pi / 180)
        return x_modified, y_modified, heading

    def run(self):
        state_gps = OrderedDict()
        state_gps['GaussX'] = 1.75   # intersection coordinate [m]
        state_gps['GaussY'] = -40   # intersection coordinate [m]
        state_gps['Heading'] = 90.  # intersection coordinate [deg]

        state_gps['GpsSpeed'] = 0          # [m/s]
        state_gps['NorthVelocity'] = 0     # [m/s]
        state_gps['EastVelocity'] = 0      # [m/s]

        state_gps['YawRate'] = 0           # [rad/s]
        state_gps['LongitudinalAcc'] = 0   # [m/s^2]
        state_gps['LateralAcc'] = 0        # [m/s^2]
        state_gps['Longitude'] = 0
        state_gps['Latitude'] = 0

        time_receive_gps = 0

        while True:
            try:
                data_gps = self.socket_gps.recv(zmq.NOBLOCK).decode('utf-8')
                GpsJson = json.loads(data_gps)
                if GpsJson is not None:
                    x_rear_axle = GpsJson["Gps"]["Gps"]["GaussX"]
                    y_rear_axle = 0.1+GpsJson["Gps"]["Gps"]["GaussY"]
                    heading = 1. + GpsJson["Gps"]["Gps"]["Heading"]
                    state_gps['GaussX'], state_gps['GaussY'], state_gps['Heading'] =\
                        self.rotate_and_move(x_rear_axle, y_rear_axle, heading)
                    if -180 < state_gps['Heading'] < -90:
                        state_gps['Heading'] += 360.
                        # state_gps['Heading'] = np.clip(state_gps['Heading'], 0, 178)

                    state_gps['NorthVelocity'] = GpsJson["Gps"]["Gps"]["NorthVelocity"]
                    state_gps['EastVelocity'] = GpsJson["Gps"]["Gps"]["EastVelocity"]
                    state_gps['GpsSpeed'] = math.sqrt(state_gps['NorthVelocity'] ** 2 + state_gps['EastVelocity']**2)

                    state_gps['YawRate'] = -GpsJson["Gps"]["Gps"]["YawRate"] * np.pi / 180
                    state_gps['LongitudinalAcc'] = GpsJson["Gps"]["Gps"]["LongitudinalAcc"]
                    state_gps['LateralAcc'] = GpsJson["Gps"]["Gps"]["LateralAcc"]
                    state_gps['Longitude'] = GpsJson["Gps"]["Gps"]["Longitude"]
                    state_gps['Latitude'] = GpsJson["Gps"]["Gps"]["Latitude"]
                    time_receive_gps = time.time() - self.time_start_gps
                    self.time_start_gps = time.time()
            except zmq.ZMQError:
                pass
            self.receive_index += 1

            with self.lock:
                self.shared_list[0] = state_gps.copy()
                self.shared_list[1] = time_receive_gps
            self.receive_index_shared.value += 1


if __name__ == '__main__':
    pass
