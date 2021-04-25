#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/28
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: subscriber_can.py
# =====================================

import json
import time
from collections import OrderedDict

import zmq


class SubscriberCan():
    def __init__(self, shared_list, receive_index, lock):
        self.shared_list = shared_list
        self.receive_index_shared = receive_index
        self.time_start_can = 0.
        self.lock = lock
        self.position_index = 0
        self.x_bias = 1.4

        context = zmq.Context()
        self.socket_can = context.socket(zmq.SUB)
        self.socket_can.connect("tcp://127.0.0.1:7777")  # 上车
        self.socket_can.setsockopt(zmq.SUBSCRIBE, "".encode('utf-8'))  # 接收所有消息

    def run(self):
        state_can = OrderedDict()
        state_can['VehicleSPeedAct'] = 0  # [m/s]
        state_can['SteerAngleAct'] = 0    # [m/s]
        state_can['AutoGear'] = 0
        state_can['VehicleMode'] = 0      # 0: manual driving; 1: autonomous driving
        state_can['Throttle'] = 0         # [deg]
        state_can['BrkOn'] = 0            # brake signal, is 1 if sending a minus deceleration
        time_receive_can = 0

        while True:
            try:
                data_can = self.socket_can.recv(zmq.NOBLOCK).decode('utf-8')
                if data_can != b'null\n':
                    CanJson = json.loads(data_can)
                    if CanJson["CAN"]["VehicleStatues"]["IsValid"] == True:
                        # todo: 车辆实际执行的动作大小，由于车辆的物理响应，实际执行动作和网络输出动作可能不同，也可能存在时延
                        state_can['VehicleSPeedAct'] = CanJson["CAN"]["VehicleStatues"]["VehicleSpeedAct"]
                        # todo: 车轮转角的零位存在偏差，单位（°），实际需要测试标定，方法是给车发送0的控制量，看看实际反馈是多少
                        state_can['SteerAngleAct'] = -1.7 + CanJson["CAN"]["VehicleStatues"]["SteerAngleAct"]

                        state_can['AutoGear'] = CanJson["CAN"]["VehicleStatues"]["AutoGear"]
                        state_can['VehicleMode'] = CanJson["CAN"]["VehicleStatues"]["VehicleMode"]
                        state_can['Throttle'] = CanJson["CAN"]["VehicleStatues"]["Throttle"]
                        state_can['BrkOn'] = CanJson["CAN"]["VehicleStatues"]["BrkOn"]
                        time_receive_can = time.time() - self.time_start_can
                        self.time_start_can = time.time()
            except zmq.ZMQError:
                pass

            with self.lock:
                self.shared_list[2] = state_can.copy()
                self.shared_list[3] = time_receive_can

            self.receive_index_shared.value += 1


if __name__ == '__main__':
    pass