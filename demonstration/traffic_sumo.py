#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: traffic.py
# =====================================

import copy
import math
import os
import random
import sys
import time
from collections import defaultdict
from math import fabs, cos, sin, pi
from utils.endtoend_env_utils import TASK2ROUTEID, L, W

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary
import traci
from traci.exceptions import FatalTraCIError
from utils.endtoend_env_utils import shift_and_rotate_coordination, _convert_car_coord_to_sumo_coord, \
    _convert_sumo_coord_to_car_coord, xy2_edgeID_lane, SUMOCFG_DIR

SUMO_BINARY = checkBinary('sumo')
SIM_PERIOD = 1.0 / 10


class Traffic(object):
    def __init__(self, shared_list, lock, step_length, mode, model_only_test, training_task='left'):  # mode 'display' or 'training'
        self.shared_list = shared_list
        self.model_only_test = model_only_test
        self.trigger = False
        self.lock = lock
        self.random_traffic = None
        self.sim_time = 0
        self.n_ego_vehicles = defaultdict(list)
        self.step_length = step_length
        self.step_time_str = str(float(step_length) / 1000)
        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None
        self.v_light = None
        self.n_ego_dict = {}
        self.mode = mode
        self.training_light_phase = 0
        self.training_task = training_task
        task2route = {'left': 'dl', 'straight': 'du', 'right': 'dr'}
        self.ego_route = task2route[self.training_task]
        try:
            traci.start(
                [SUMO_BINARY, "-c", SUMOCFG_DIR,
                 "--step-length", self.step_time_str,
                 "--lateral-resolution", "3.5",
                 "--random",
                 # "--start",
                 # "--quit-on-end",
                 "--no-warnings",
                 "--no-step-log",
                 # '--seed', str(int(seed))
                 ], numRetries=5)  # '--seed', str(int(seed))
        except FatalTraCIError:
            print('Retry by other port')
            port = sumolib.miscutils.getFreeSocketPort()
            traci.start(
                [SUMO_BINARY, "-c", SUMOCFG_DIR,
                 "--step-length", self.step_time_str,
                 "--lateral-resolution", "3.5",
                 "--random",
                 # "--start",
                 # "--quit-on-end",
                 "--no-warnings",
                 "--no-step-log",
                 # '--seed', str(int(seed))
                 ], port=port, numRetries=5)  # '--seed', str(int(seed))

        traci.vehicle.subscribeContext('collector',
                                       traci.constants.CMD_GET_VEHICLE_VARIABLE,
                                       999999, [traci.constants.VAR_POSITION,
                                                traci.constants.VAR_LENGTH,
                                                traci.constants.VAR_WIDTH,
                                                traci.constants.VAR_ANGLE,
                                                traci.constants.VAR_SIGNALS,
                                                traci.constants.VAR_SPEED,
                                                # traci.constants.VAR_TYPE,
                                                # traci.constants.VAR_EMERGENCY_DECEL,
                                                # traci.constants.VAR_LANE_INDEX,
                                                # traci.constants.VAR_LANEPOSITION,
                                                traci.constants.VAR_EDGES,
                                                # traci.constants.VAR_ROUTE_INDEX
                                                ],
                                       0, 2147483647)
        while traci.simulation.getTime() < 100:  # related with step-length
            if traci.simulation.getTime() < 80:  # related with step-length
                traci.trafficlight.setPhase('0', 2)
            else:
                traci.trafficlight.setPhase('0', 0)
            traci.simulationStep()

    def __del__(self):
        traci.close()

    def add_self_car(self, n_ego_dict):
        for egoID, ego_dict in n_ego_dict.items():
            ego_v_x = ego_dict['v_x']
            ego_v_y = ego_dict['v_y']
            ego_l = ego_dict['l']
            ego_x = ego_dict['x']
            ego_y = ego_dict['y']
            ego_phi = ego_dict['phi']
            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi, ego_l)
            edgeID, lane = xy2_edgeID_lane(ego_x, ego_y)
            try:
                traci.vehicle.remove(vehID=egoID)
            except traci.exceptions.TraCIException:
                pass
            # traci.simulationStep()
            traci.vehicle.addLegacy(vehID=egoID, routeID=ego_dict['routeID'],
                                    #depart=0, pos=20, lane=3, speed=ego_dict['v_x'],
                                    typeID='self_car')
            # if random.random() > 0.5:  # todo: use for resolve other vehicles waiting ego, not always useful
            #     traci.vehicle.setRouteID(egoID, 'dr')
            # else:
            #     traci.vehicle.setRouteID(egoID, self.ego_route)
            traci.vehicle.moveToXY(egoID, edgeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keepRoute=1)
            traci.vehicle.setLength(egoID, ego_dict['l'])
            traci.vehicle.setWidth(egoID, ego_dict['w'])
            traci.vehicle.setSpeed(egoID, math.sqrt(ego_v_x ** 2 + ego_v_y ** 2))

    def generate_random_traffic(self):
        random_traffic = traci.vehicle.getContextSubscriptionResults('collector')
        random_traffic = copy.deepcopy(random_traffic)

        for ego_id in self.n_ego_dict.keys():
            if ego_id in random_traffic:
                del random_traffic[ego_id]

        return random_traffic

    def init_traffic(self, init_n_ego_dict):
        self.sim_time = 0
        self.n_ego_vehicles = defaultdict(list)
        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None
        self.v_light = None
        self.training_light_phase = 0
        if self.training_task == 'right':
            if random.random() > 0.5:
                self.training_light_phase = 2
        self.n_ego_dict = init_n_ego_dict
        traci.trafficlight.setPhase('0', self.training_light_phase)
        random_traffic = self.generate_random_traffic()

        self.add_self_car(init_n_ego_dict)

        # move ego to the given position and remove conflict cars
        for egoID, ego_dict in self.n_ego_dict.items():
            ego_x, ego_y, ego_v_x, ego_v_y, ego_phi, ego_l, ego_w = ego_dict['x'], ego_dict['y'], ego_dict['v_x'],\
                                                                    ego_dict['v_y'], ego_dict['phi'], ego_dict['l'], \
                                                                    ego_dict['w']
            for veh in random_traffic:
                x_in_sumo, y_in_sumo = random_traffic[veh][traci.constants.VAR_POSITION]
                a_in_sumo = random_traffic[veh][traci.constants.VAR_ANGLE]
                veh_l = random_traffic[veh][traci.constants.VAR_LENGTH]
                veh_v = random_traffic[veh][traci.constants.VAR_SPEED]
                # veh_sig = random_traffic[veh][traci.constants.VAR_SIGNALS]
                # 10: left and brake 9: right and brake 1: right 8: brake 0: no signal 2: left

                x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, veh_l)
                x_in_ego_coord, y_in_ego_coord, a_in_ego_coord = shift_and_rotate_coordination(x, y, a, ego_x,
                                                                                               ego_y, ego_phi)
                ego_x_in_veh_coord, ego_y_in_veh_coord, ego_a_in_veh_coord = shift_and_rotate_coordination(0, 0, 0,
                                                                                                           x_in_ego_coord,
                                                                                                           y_in_ego_coord,
                                                                                                           a_in_ego_coord)
                if (-5 < x_in_ego_coord < 1 * (ego_v_x) + ego_l/2. + veh_l/2. + 2 and abs(y_in_ego_coord) < 3) or \
                        (-5 < ego_x_in_veh_coord < 1 * (veh_v) + ego_l/2. + veh_l/2. + 2 and abs(ego_y_in_veh_coord) <3):
                    traci.vehicle.moveToXY(veh, '4i', 1, -80, 1.85, 180, 2) #TODO: check
                    # traci.vehicle.remove(vehID=veh)
                # if 0<x_in_sumo<3.5 and -22<y_in_sumo<-15:# and veh_sig!=1 and veh_sig!=9:
                #     traci.vehicle.moveToXY(veh, '4o', 1, -80, 1.85, 180,2)
                #     traci.vehicle.remove(vehID=veh)

    def _get_vehicles(self):
        self.n_ego_vehicles = defaultdict(list)
        veh_infos = traci.vehicle.getContextSubscriptionResults('collector')
        for egoID in self.n_ego_dict.keys():
            veh_info_dict = copy.deepcopy(veh_infos)
            for i, veh in enumerate(veh_info_dict):
                if veh != egoID:
                    length = veh_info_dict[veh][traci.constants.VAR_LENGTH]
                    width = veh_info_dict[veh][traci.constants.VAR_WIDTH]
                    route = veh_info_dict[veh][traci.constants.VAR_EDGES]
                    if route[0] == '4i':
                        continue
                    x_in_sumo, y_in_sumo = veh_info_dict[veh][traci.constants.VAR_POSITION]
                    a_in_sumo = veh_info_dict[veh][traci.constants.VAR_ANGLE]
                    # transfer x,y,a in car coord
                    x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, length)
                    v = veh_info_dict[veh][traci.constants.VAR_SPEED]
                    self.n_ego_vehicles[egoID].append(dict(x=x, y=y, v=v, phi=a, l=length,
                                                           w=width, route=route))

    def _get_traffic_light(self):
        self.v_light = traci.trafficlight.getPhase('0')

    def sim_step(self):
        self.sim_time += SIM_PERIOD
        if self.mode == 'training':
            traci.trafficlight.setPhase('0', self.training_light_phase)
        traci.simulationStep()
        self._get_vehicles()
        self._get_traffic_light()
        self.collision_check()
        for egoID, collision_flag in self.n_ego_collision_flag.items():
            if collision_flag:
                self.collision_flag = True
                self.collision_ego_id = egoID

    def set_own_car(self, n_ego_dict_):
        assert len(self.n_ego_dict) == len(n_ego_dict_)
        for egoID in self.n_ego_dict.keys():
            self.n_ego_dict[egoID]['v_x'] = ego_v_x = n_ego_dict_[egoID]['v_x']
            self.n_ego_dict[egoID]['v_y'] = ego_v_y = n_ego_dict_[egoID]['v_y']
            self.n_ego_dict[egoID]['r'] = ego_r = n_ego_dict_[egoID]['r']
            self.n_ego_dict[egoID]['x'] = ego_x = n_ego_dict_[egoID]['x']
            self.n_ego_dict[egoID]['y'] = ego_y = n_ego_dict_[egoID]['y']
            self.n_ego_dict[egoID]['phi'] = ego_phi = n_ego_dict_[egoID]['phi']

            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi,
                                                                                           self.n_ego_dict[egoID]['l'])
            egdeID, lane = xy2_edgeID_lane(ego_x, ego_y)
            keeproute = 1
            # if self.training_task == 'left':  # TODO
            #     keeproute = 2 if ego_x > 0 and ego_y > -7 else 1
            try:
                traci.vehicle.moveToXY(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
            except traci.exceptions.TraCIException:
                print(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
                traci.vehicle.moveToXY(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
            traci.vehicle.setSpeed(egoID, math.sqrt(ego_v_x**2+ego_v_y**2))

    def collision_check(self):  # True: collision
        flag_dict = dict()
        for egoID, list_of_veh_dict in self.n_ego_vehicles.items():
            ego_x = self.n_ego_dict[egoID]['x']
            ego_y = self.n_ego_dict[egoID]['y']
            ego_phi = self.n_ego_dict[egoID]['phi']
            ego_l = self.n_ego_dict[egoID]['l']
            ego_w = self.n_ego_dict[egoID]['w']
            ego_lw = (ego_l - ego_w) / 2
            ego_x0 = (ego_x + cos(ego_phi / 180 * pi) * ego_lw)
            ego_y0 = (ego_y + sin(ego_phi / 180 * pi) * ego_lw)
            ego_x1 = (ego_x - cos(ego_phi / 180 * pi) * ego_lw)
            ego_y1 = (ego_y - sin(ego_phi / 180 * pi) * ego_lw)
            flag_dict[egoID] = False

            for veh in list_of_veh_dict:
                if fabs(veh['x'] - ego_x) < 10 and fabs(veh['y'] - ego_y) < 10:
                    surrounding_lw = (veh['l'] - veh['w']) / 2
                    surrounding_x0 = (veh['x'] + cos(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_y0 = (veh['y'] + sin(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_x1 = (veh['x'] - cos(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_y1 = (veh['y'] - sin(veh['phi'] / 180 * pi) * surrounding_lw)
                    collision_check_dis = ((veh['w'] + ego_w) / 2 + 0.5) ** 2
                    if (ego_x0 - surrounding_x0) ** 2 + (ego_y0 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x0 - surrounding_x1) ** 2 + (ego_y0 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x1) ** 2 + (ego_y1 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x0) ** 2 + (ego_y1 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True

        self.n_ego_collision_flag = flag_dict

    def is_triggered(self, model_only_test, vehicle_mode):
        if model_only_test:
            self.trigger = True
            print('model only test, traffic is triggered directly')
        elif vehicle_mode == 1:
            self.trigger = True
            print('switch to automated driving, traffic is triggered')
        else:
            self.trigger = False
            print('traffic is not triggered')

    def run(self):
        start_time = time.time()
        state_other = []
        v_light = 0
        while True:
            time.sleep(self.step_length/1000-0.01)
            state_ego = self.shared_list[9]
            if isinstance(state_ego, dict):
                ego_x = state_ego['GaussX']  # intersection coordinate [m]
                ego_y = state_ego['GaussY']  # intersection coordinate [m]
                ego_phi = state_ego['Heading']  # intersection coordinate [deg]
                ego_v = state_ego['GpsSpeed']
                out = dict(v_x=ego_v, v_y=0, r=0, x=ego_x, y=ego_y,
                           phi=ego_phi, l=L, w=W, routeID=TASK2ROUTEID[self.training_task])
                if not self.trigger:
                    self.is_triggered(self.model_only_test, state_ego['VehicleMode'])
                    if self.trigger:
                        self.init_traffic(dict(ego=out))
                        self._get_vehicles()
                else:
                    self.set_own_car(dict(ego=out))
                    self.sim_step()
                    v_light = self.v_light
                state_other = self.n_ego_vehicles['ego']
            delta_time = time.time() - start_time
            start_time = time.time()

            with self.lock:
                self.shared_list[4] = state_other.copy()
                self.shared_list[5] = delta_time
                self.shared_list[13] = v_light


