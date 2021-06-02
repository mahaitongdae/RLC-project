#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: endtoend.py
# =====================================

import warnings
from collections import OrderedDict
from math import cos, sin, pi

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import logical_and
from gym.utils import seeding

# gym.envs.user_defined.toyota_env.
from dynamics_and_models import VehicleDynamics, ReferencePath, EnvironmentModel
from endtoend_env_utils import shift_coordination, rotate_coordination, rotate_and_shift_coordination, deal_with_phi, \
    L, W, CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER, judge_feasible, MODE2TASK, VEHICLE_MODE_DICT, VEH_NUM, EXPECTED_V
from traffic import Traffic

warnings.filterwarnings("ignore")


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = gym.spaces.Box(low, high, dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class CrossroadEnd2end(gym.Env):
    def __init__(self,
                 training_task,  # 'left', 'straight', 'right'
                 num_future_data=0,
                 display=False,
                 **kwargs):
        metadata = {'render.modes': ['human']}
        self.dynamics = VehicleDynamics()
        self.interested_vehs = None
        self.training_task = training_task
        self.ref_path = ReferencePath(self.training_task, **kwargs)
        self.detected_vehicles = None
        self.all_vehicles = None
        self.ego_dynamics = None
        self.num_future_data = num_future_data
        self.env_model = EnvironmentModel(training_task, num_future_data)
        self.init_state = {}
        self.action_number = 2
        self.exp_v = EXPECTED_V #TODO: temp
        self.ego_l, self.ego_w = L, W
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_number,), dtype=np.float32)

        self.seed()
        self.v_light = None
        self.step_length = 100  # ms

        self.step_time = self.step_length / 1000.0
        self.init_state = self._reset_init_state()
        self.obs = None
        self.action = None
        self.veh_mode_dict = VEHICLE_MODE_DICT[self.training_task]
        self.veh_num = VEH_NUM[self.training_task]
        self.virtual_red_light_vehicle = False

        self.done_type = 'not_done_yet'
        self.reward_info = None
        self.ego_info_dim = None
        self.per_tracking_info_dim = None
        self.per_veh_info_dim = None
        if not display:
            self.traffic = Traffic(self.step_length,
                                   mode='training',
                                   init_n_ego_dict=self.init_state,
                                   training_task=self.training_task)
            self.reset()
            action = self.action_space.sample()
            observation, _reward, done, _info = self.step(action)
            self._set_observation_space(observation)
            plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):  # kwargs include three keys
        self.ref_path = ReferencePath(self.training_task, **kwargs)
        self.init_state = self._reset_init_state()
        self.traffic.init_traffic(self.init_state)
        self.traffic.sim_step()
        ego_dynamics = self._get_ego_dynamics([self.init_state['ego']['v_x'],
                                               self.init_state['ego']['v_y'],
                                               self.init_state['ego']['r'],
                                               self.init_state['ego']['x'],
                                               self.init_state['ego']['y'],
                                               self.init_state['ego']['phi']],
                                              [0,
                                               0,
                                               self.dynamics.vehicle_params['miu'],
                                               self.dynamics.vehicle_params['miu']]
                                              )
        self._get_all_info(ego_dynamics)
        self.obs = self._get_obs()
        self.action = None
        self.reward_info = None
        self.done_type = 'not_done_yet'
        if np.random.random() > 0.9:
            self.virtual_red_light_vehicle = True
        else:
            self.virtual_red_light_vehicle = False
        return self.obs

    def close(self):
        del self.traffic

    def step(self, action):
        self.action = self._action_transformation_for_end2end(action)
        reward, self.reward_info = self.compute_reward(self.obs, self.action)
        next_ego_state, next_ego_params = self._get_next_ego_state(self.action)
        ego_dynamics = self._get_ego_dynamics(next_ego_state, next_ego_params)
        self.traffic.set_own_car(dict(ego=ego_dynamics))
        self.traffic.sim_step()
        all_info = self._get_all_info(ego_dynamics)
        self.obs = self._get_obs()
        self.done_type, done = self._judge_done()
        self.reward_info.update({'final_rew': reward})
        all_info.update({'reward_info': self.reward_info, 'ref_index': self.ref_path.ref_index})
        return self.obs, reward, done, all_info

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_ego_dynamics(self, next_ego_state, next_ego_params):
        out = dict(v_x=next_ego_state[0],
                   v_y=next_ego_state[1],
                   r=next_ego_state[2],
                   x=next_ego_state[3],
                   y=next_ego_state[4],
                   phi=next_ego_state[5],
                   l=self.ego_l,
                   w=self.ego_w,
                   alpha_f=next_ego_params[0],
                   alpha_r=next_ego_params[1],
                   miu_f=next_ego_params[2],
                   miu_r=next_ego_params[3],)
        miu_f, miu_r = out['miu_f'], out['miu_r']
        F_zf, F_zr = self.dynamics.vehicle_params['F_zf'], self.dynamics.vehicle_params['F_zr']
        C_f, C_r = self.dynamics.vehicle_params['C_f'], self.dynamics.vehicle_params['C_r']
        alpha_f_bound, alpha_r_bound = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bound = miu_r * self.dynamics.vehicle_params['g'] / (abs(out['v_x'])+1e-8)

        l, w, x, y, phi = out['l'], out['w'], out['x'], out['y'], out['phi']

        def cal_corner_point_of_ego_car():
            x0, y0, a0 = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -phi)
            x1, y1, a1 = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -phi)
            x2, y2, a2 = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -phi)
            x3, y3, a3 = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -phi)
            return (x0, y0), (x1, y1), (x2, y2), (x3, y3)
        Corner_point = cal_corner_point_of_ego_car()
        out.update(dict(alpha_f_bound=alpha_f_bound,
                        alpha_r_bound=alpha_r_bound,
                        r_bound=r_bound,
                        Corner_point=Corner_point))

        return out

    def _get_all_info(self, ego_dynamics):  # used to update info, must be called every timestep before _get_obs
        # to fetch info
        self.all_vehicles = self.traffic.n_ego_vehicles['ego']  # coordination 2
        self.ego_dynamics = ego_dynamics  # coordination 2
        self.v_light = self.traffic.v_light

        # all_vehicles
        # dict(x=x, y=y, v=v, phi=a, l=length,
        #      w=width, route=route)

        all_info = dict(all_vehicles=self.all_vehicles,
                        ego_dynamics=self.ego_dynamics,
                        v_light=self.v_light)
        return all_info

    def _judge_done(self):
        """
        :return:
         1: bad done: collision
         2: bad done: break_road_constrain
         3: good done: task succeed
         4: not done
        """
        if self.traffic.collision_flag:
            return 'collision', 1
        if self._break_road_constrain():
            return 'break_road_constrain', 1
        elif self._deviate_too_much():
            return 'deviate_too_much', 1
        elif self._break_stability():
            return 'break_stability', 1
        elif self._break_red_light():
            return 'break_red_light', 1
        elif self._is_achieve_goal():
            return 'good_done', 1
        else:
            return 'not_done_yet', 0

    def _deviate_too_much(self):
        delta_y, delta_phi, delta_v = self.obs[self.ego_info_dim:self.ego_info_dim+3]
        return True if abs(delta_y) > 15 else False

    def _break_road_constrain(self):
        results = list(map(lambda x: judge_feasible(*x, self.training_task), self.ego_dynamics['Corner_point']))
        return not all(results)

    def _break_stability(self):
        alpha_f, alpha_r, miu_f, miu_r = self.ego_dynamics['alpha_f'], self.ego_dynamics['alpha_r'], \
                                         self.ego_dynamics['miu_f'], self.ego_dynamics['miu_r']
        alpha_f_bound, alpha_r_bound = self.ego_dynamics['alpha_f_bound'], self.ego_dynamics['alpha_r_bound']
        r_bound = self.ego_dynamics['r_bound']
        # if -alpha_f_bound < alpha_f < alpha_f_bound \
        #         and -alpha_r_bound < alpha_r < alpha_r_bound and \
        #         -r_bound < self.ego_dynamics['r'] < r_bound:
        if -r_bound < self.ego_dynamics['r'] < r_bound:
            return False
        else:
            return True

    def _break_red_light(self):
        return True if self.v_light != 0 and self.ego_dynamics['y'] > -CROSSROAD_SIZE/2 and self.training_task != 'right' else False

    def _is_achieve_goal(self):
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        if self.training_task == 'left':
            return True if x < -CROSSROAD_SIZE/2 - 10 and 0 < y < LANE_NUMBER*LANE_WIDTH else False
        elif self.training_task == 'right':
            return True if x > CROSSROAD_SIZE/2 + 10 and -LANE_NUMBER*LANE_WIDTH < y < 0 else False
        else:
            assert self.training_task == 'straight'
            return True if y > CROSSROAD_SIZE/2 + 10 and 0 < x < LANE_NUMBER*LANE_WIDTH else False

    def _action_transformation_for_end2end(self, action):  # [-1, 1]
        action = np.clip(action, -1.05, 1.05)
        steer_norm, a_x_norm = action[0], action[1]
        scaled_steer = 0.4 * steer_norm
        scaled_a_x = 2.25*a_x_norm - 0.75  # [-3, 1.5]
        scaled_action = np.array([scaled_steer, scaled_a_x], dtype=np.float32)
        return scaled_action

    def _get_next_ego_state(self, trans_action):
        current_v_x = self.ego_dynamics['v_x']
        current_v_y = self.ego_dynamics['v_y']
        current_r = self.ego_dynamics['r']
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_phi = self.ego_dynamics['phi']
        steer, a_x = trans_action
        state = np.array([[current_v_x, current_v_y, current_r, current_x, current_y, current_phi]], dtype=np.float32)
        action = np.array([[steer, a_x]], dtype=np.float32)
        next_ego_state, next_ego_params = self.dynamics.prediction(state, action, 10)
        next_ego_state, next_ego_params = next_ego_state.numpy()[0],  next_ego_params.numpy()[0]
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.
        next_ego_state[-1] = deal_with_phi(next_ego_state[-1])
        return next_ego_state, next_ego_params

    def _get_obs(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_v_x = self.ego_dynamics['v_x']

        vehs_vector = self._construct_veh_vector_short(exit_)
        ego_vector = self._construct_ego_vector_short()
        tracking_error = self.ref_path.tracking_error_vector(np.array([ego_x], dtype=np.float32),
                                                             np.array([ego_y], dtype=np.float32),
                                                             np.array([ego_phi], dtype=np.float32),
                                                             np.array([ego_v_x], dtype=np.float32),
                                                             self.num_future_data).numpy()[0]
        self.per_tracking_info_dim = 3

        vector = np.concatenate((ego_vector, tracking_error, vehs_vector), axis=0)
        # vector = self.convert_vehs_to_rela(vector)

        return vector

    # def convert_vehs_to_rela(self, obs_abso):
    #     ego_infos, tracking_infos, veh_infos = obs_abso[:self.ego_info_dim], \
    #                                            obs_abso[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                      self.num_future_data + 1)], \
    #                                            obs_abso[self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                        self.num_future_data + 1):]
    #     ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi = ego_infos
    #     ego = np.array([ego_x, ego_y, 0, 0]*int(len(veh_infos)/self.per_veh_info_dim), dtype=np.float32)
    #     vehs_rela = veh_infos - ego
    #     out = np.concatenate((ego_infos, tracking_infos, vehs_rela), axis=0)
    #     return out
    #
    # def convert_vehs_to_abso(self, obs_rela):
    #     ego_infos, tracking_infos, veh_rela = obs_rela[:self.ego_info_dim], \
    #                                            obs_rela[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                    self.num_future_data + 1)], \
    #                                            obs_rela[self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                    self.num_future_data + 1):]
    #     ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi = ego_infos
    #     ego = np.array([ego_x, ego_y, 0, 0]*int(len(veh_rela)/self.per_veh_info_dim), dtype=np.float32)
    #     vehs_abso = veh_rela + ego
    #     out = np.concatenate((ego_infos, tracking_infos, vehs_abso), axis=0)
    #     return out

    def _construct_ego_vector_short(self):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi]
        self.ego_info_dim = 6
        return np.array(ego_feature, dtype=np.float32)

    def _construct_veh_vector_short(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        v_light = self.v_light
        vehs_vector = []

        name_settings = dict(D=dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i'),
                             R=dict(do='2o', di='2i', ro='3o', ri='3i', uo='4o', ui='4i', lo='1o', li='1i'),
                             U=dict(do='3o', di='3i', ro='4o', ri='4i', uo='1o', ui='1i', lo='2o', li='2i'),
                             L=dict(do='4o', di='4i', ro='1o', ri='1i', uo='2o', ui='2i', lo='3o', li='3i'))

        name_setting = name_settings[exit_]

        def filter_interested_vehicles(vs, task):
            dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld = [], [], [], [], [], [], [], [], [], [], [], []
            for v in vs:
                route_list = v['route']
                start = route_list[0]
                end = route_list[1]
                if start == name_setting['do'] and end == name_setting['li']:
                    dl.append(v)
                elif start == name_setting['do'] and end == name_setting['ui']:
                    du.append(v)
                elif start == name_setting['do'] and end == name_setting['ri']:
                    dr.append(v)

                elif start == name_setting['ro'] and end == name_setting['di']:
                    rd.append(v)
                elif start == name_setting['ro'] and end == name_setting['li']:
                    rl.append(v)
                elif start == name_setting['ro'] and end == name_setting['ui']:
                    ru.append(v)

                elif start == name_setting['uo'] and end == name_setting['ri']:
                    ur.append(v)
                elif start == name_setting['uo'] and end == name_setting['di']:
                    ud.append(v)
                elif start == name_setting['uo'] and end == name_setting['li']:
                    ul.append(v)

                elif start == name_setting['lo'] and end == name_setting['ui']:
                    lu.append(v)
                elif start == name_setting['lo'] and end == name_setting['ri']:
                    lr.append(v)
                elif start == name_setting['lo'] and end == name_setting['di']:
                    ld.append(v)
            if self.training_task != 'right':
                if (v_light != 0 and ego_y < -CROSSROAD_SIZE/2) \
                        or (self.virtual_red_light_vehicle and ego_y < -CROSSROAD_SIZE/2):
                    dl.append(dict(x=LANE_WIDTH/2, y=-CROSSROAD_SIZE/2+2.5, v=0., phi=90, l=5, w=2.5, route=None))
                    du.append(dict(x=LANE_WIDTH*1.5, y=-CROSSROAD_SIZE/2+2.5, v=0., phi=90, l=5, w=2.5, route=None))

            # fetch veh in range
            dl = list(filter(lambda v: v['x'] > -CROSSROAD_SIZE/2-10 and v['y'] > ego_y-2, dl))  # interest of left straight
            du = list(filter(lambda v: ego_y-2 < v['y'] < CROSSROAD_SIZE/2+10 and v['x'] < ego_x+5, du))  # interest of left straight

            dr = list(filter(lambda v: v['x'] < CROSSROAD_SIZE/2+10 and v['y'] > ego_y, dr))  # interest of right

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(lambda v: v['x'] < CROSSROAD_SIZE/2+10 and v['y'] < CROSSROAD_SIZE/2+10, ru))  # interest of straight

            if task == 'straight':
                ur = list(filter(lambda v: v['x'] < ego_x + 7 and ego_y < v['y'] < CROSSROAD_SIZE/2+10, ur))  # interest of straight
            elif task == 'right':
                ur = list(filter(lambda v: v['x'] < CROSSROAD_SIZE/2+10 and v['y'] < CROSSROAD_SIZE/2, ur))  # interest of right
            ud = list(filter(lambda v: max(ego_y-2, -CROSSROAD_SIZE/2) < v['y'] < CROSSROAD_SIZE/2 and ego_x > v['x'], ud))  # interest of left
            ul = list(filter(lambda v: -CROSSROAD_SIZE/2-10 < v['x'] < ego_x and v['y'] < CROSSROAD_SIZE/2, ul))  # interest of left

            lu = lu  # not interest in case of traffic light
            lr = list(filter(lambda v: -CROSSROAD_SIZE/2-10 < v['x'] < CROSSROAD_SIZE/2+10, lr))  # interest of right
            ld = ld  # not interest in case of traffic light

            # sort
            dl = sorted(dl, key=lambda v: (v['y'], -v['x']))
            du = sorted(du, key=lambda v: v['y'])
            dr = sorted(dr, key=lambda v: (v['y'], v['x']))

            ru = sorted(ru, key=lambda v: (-v['x'], v['y']), reverse=True)

            if task == 'straight':
                ur = sorted(ur, key=lambda v: v['y'])
            elif task == 'right':
                ur = sorted(ur, key=lambda v: (-v['y'], v['x']), reverse=True)

            ud = sorted(ud, key=lambda v: v['y'])
            ul = sorted(ul, key=lambda v: (-v['y'], -v['x']), reverse=True)

            lr = sorted(lr, key=lambda v: -v['x'])

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                if len(sorted_list) >= num:
                    return sorted_list[:num]
                else:
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                    return sorted_list

            mode2fillvalue = dict(
                dl=dict(x=LANE_WIDTH/2, y=-(CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '4i')),
                du=dict(x=LANE_WIDTH*1.5, y=-(CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '3i')),
                dr=dict(x=LANE_WIDTH*(LANE_NUMBER-0.5), y=-(CROSSROAD_SIZE/2+30), v=0, phi=90, w=2.5, l=5, route=('1o', '2i')),
                ru=dict(x=(CROSSROAD_SIZE/2+15), y=LANE_WIDTH*(LANE_NUMBER-0.5), v=0, phi=180, w=2.5, l=5, route=('2o', '3i')),
                ur=dict(x=-LANE_WIDTH/2, y=(CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '2i')),
                ud=dict(x=-LANE_WIDTH*1.5, y=(CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '1i')),
                ul=dict(x=-LANE_WIDTH*(LANE_NUMBER-0.5), y=(CROSSROAD_SIZE/2+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '4i')),
                lr=dict(x=-(CROSSROAD_SIZE/2+20), y=-LANE_WIDTH*1.5, v=0, phi=0, w=2.5, l=5, route=('4o', '2i')))

            tmp = OrderedDict()
            for mode, num in VEHICLE_MODE_DICT[task].items():
                tmp[mode] = slice_or_fill(eval(mode), mode2fillvalue[mode], num)

            return tmp

        list_of_interested_veh_dict = []
        self.interested_vehs = filter_interested_vehicles(self.all_vehicles, self.training_task)
        for part in list(self.interested_vehs.values()):
            list_of_interested_veh_dict.extend(part)

        for veh in list_of_interested_veh_dict:
            veh_x, veh_y, veh_v, veh_phi = veh['x'], veh['y'], veh['v'], veh['phi']
            vehs_vector.extend([veh_x, veh_y, veh_v, veh_phi])
        self.per_veh_info_dim = 4
        return np.array(vehs_vector, dtype=np.float32)

    def recover_orig_position_fn(self, transformed_x, transformed_y, x, y, d):  # x, y, d are used to transform
        # coordination
        transformed_x, transformed_y, _ = rotate_coordination(transformed_x, transformed_y, 0, -d)
        orig_x, orig_y = shift_coordination(transformed_x, transformed_y, -x, -y)
        return orig_x, orig_y

    def _reset_init_state(self):
        if self.training_task == 'left':
            random_index = int(np.random.random()*(900+500)) + 700
        elif self.training_task == 'straight':
            random_index = int(np.random.random()*(1200+500)) + 700
        else:
            random_index = int(np.random.random()*(420+500)) + 700

        x, y, phi = self.ref_path.indexs2points(random_index)
        # v = 7 + 6 * np.random.random()
        v = EXPECTED_V * np.random.random()
        if self.training_task == 'left':
            routeID = 'dl'
        elif self.training_task == 'straight':
            routeID = 'du'
        else:
            assert self.training_task == 'right'
            routeID = 'dr'
        return dict(ego=dict(v_x=v,
                             v_y=0,
                             r=0,
                             x=x.numpy(),
                             y=y.numpy(),
                             phi=phi.numpy(),
                             l=self.ego_l,
                             w=self.ego_w,
                             routeID=routeID,
                             ))

    def compute_reward(self, obs, action):
        obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
        reward, _, _, _, _, reward_dict = \
            self.env_model.compute_rewards(obses, actions)
        for k, v in reward_dict.items():
            reward_dict[k] = v.numpy()[0]
        return reward.numpy()[0], reward_dict

    def render(self, mode='human'):
        if mode == 'human':
            # plot basic map
            square_length = CROSSROAD_SIZE
            extension = 40
            lane_width = LANE_WIDTH
            light_line_width = 3
            dotted_line_style = '--'
            solid_line_style = '-'

            plt.cla()
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            ax.axis("equal")
            # ax.add_patch(plt.Rectangle((-square_length / 2 - extension, -square_length / 2 - extension),
            #                            square_length + 2 * extension, square_length + 2 * extension, edgecolor='black',
            #                            facecolor='none', linewidth=2))

            # ----------arrow--------------
            plt.arrow(lane_width/2, -square_length / 2-10, 0, 5, color='b')
            plt.arrow(lane_width/2, -square_length / 2-10+5, -0.5, 0, color='b', head_width=1)
            plt.arrow(lane_width*1.5, -square_length / 2-10, 0, 4, color='b', head_width=1)
            plt.arrow(lane_width*2.5, -square_length / 2 - 10, 0, 5, color='b')
            plt.arrow(lane_width*2.5, -square_length / 2 - 10+5, 0.5, 0, color='b', head_width=1)

            # ----------horizon--------------

            plt.plot([-square_length / 2 - extension, -square_length / 2], [0.3, 0.3], color='orange')
            plt.plot([-square_length / 2 - extension, -square_length / 2], [-0.3, -0.3], color='orange')
            plt.plot([square_length / 2 + extension, square_length / 2], [0.3, 0.3], color='orange')
            plt.plot([square_length / 2 + extension, square_length / 2], [-0.3, -0.3], color='orange')

            #
            for i in range(1, LANE_NUMBER + 1):
                linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
                linewidth = 1 if i < LANE_NUMBER else 2
                plt.plot([-square_length / 2 - extension, -square_length / 2], [i * lane_width, i * lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([square_length / 2 + extension, square_length / 2], [i * lane_width, i * lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([-square_length / 2 - extension, -square_length / 2], [-i * lane_width, -i * lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([square_length / 2 + extension, square_length / 2], [-i * lane_width, -i * lane_width],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # ----------vertical----------------
            plt.plot([0.3, 0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
            plt.plot([-0.3, -0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
            plt.plot([0.3, 0.3], [square_length / 2 + extension, square_length / 2], color='orange')
            plt.plot([-0.3, -0.3], [square_length / 2 + extension, square_length / 2], color='orange')

            #
            for i in range(1, LANE_NUMBER + 1):
                linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
                linewidth = 1 if i < LANE_NUMBER else 2
                plt.plot([i * lane_width, i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([i * lane_width, i * lane_width], [square_length / 2 + extension, square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([-i * lane_width, -i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)
                plt.plot([-i * lane_width, -i * lane_width], [square_length / 2 + extension, square_length / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # ----------stop line--------------
            # plt.plot([0, 2 * lane_width], [-square_length / 2, -square_length / 2],
            #          color='black')
            # plt.plot([-2 * lane_width, 0], [square_length / 2, square_length / 2],
            #          color='black')
            # plt.plot([-square_length / 2, -square_length / 2], [0, -2 * lane_width],
            #          color='black')
            # plt.plot([square_length / 2, square_length / 2], [2 * lane_width, 0],
            #          color='black')
            v_light = self.v_light
            if v_light == 0:
                v_color, h_color = 'green', 'red'
            elif v_light == 1:
                v_color, h_color = 'orange', 'red'
            elif v_light == 2:
                v_color, h_color = 'red', 'green'
            else:
                v_color, h_color = 'red', 'orange'

            plt.plot([0, (LANE_NUMBER-1)*lane_width], [-square_length / 2, -square_length / 2],
                     color=v_color, linewidth=light_line_width)
            plt.plot([(LANE_NUMBER-1)*lane_width, LANE_NUMBER * lane_width], [-square_length / 2, -square_length / 2],
                     color='green', linewidth=light_line_width)

            plt.plot([-LANE_NUMBER * lane_width, -(LANE_NUMBER-1)*lane_width], [square_length / 2, square_length / 2],
                     color='green', linewidth=light_line_width)
            plt.plot([-(LANE_NUMBER-1)*lane_width, 0], [square_length / 2, square_length / 2],
                     color=v_color, linewidth=light_line_width)

            plt.plot([-square_length / 2, -square_length / 2], [0, -(LANE_NUMBER-1)*lane_width],
                     color=h_color, linewidth=light_line_width)
            plt.plot([-square_length / 2, -square_length / 2], [-(LANE_NUMBER-1)*lane_width, -LANE_NUMBER * lane_width],
                     color='green', linewidth=light_line_width)

            plt.plot([square_length / 2, square_length / 2], [(LANE_NUMBER-1)*lane_width, 0],
                     color=h_color, linewidth=light_line_width)
            plt.plot([square_length / 2, square_length / 2], [LANE_NUMBER * lane_width, (LANE_NUMBER-1)*lane_width],
                     color='green', linewidth=light_line_width)

            # ----------Oblique--------------
            plt.plot([LANE_NUMBER * lane_width, square_length / 2], [-square_length / 2, -LANE_NUMBER * lane_width],
                     color='black', linewidth=2)
            plt.plot([LANE_NUMBER * lane_width, square_length / 2], [square_length / 2, LANE_NUMBER * lane_width],
                     color='black', linewidth=2)
            plt.plot([-LANE_NUMBER * lane_width, -square_length / 2], [-square_length / 2, -LANE_NUMBER * lane_width],
                     color='black', linewidth=2)
            plt.plot([-LANE_NUMBER * lane_width, -square_length / 2], [square_length / 2, LANE_NUMBER * lane_width],
                     color='black', linewidth=2)

            def is_in_plot_area(x, y, tolerance=5):
                if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
                        -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
                    return True
                else:
                    return False

            def draw_rotate_rec(x, y, a, l, w, color, linestyle='-'):
                RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
                RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
                LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
                LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
                ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color, linestyle=linestyle)
                ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color, linestyle=linestyle)
                ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color, linestyle=linestyle)
                ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color, linestyle=linestyle)

            def plot_phi_line(x, y, phi, color):
                line_length = 5
                x_forw, y_forw = x + line_length * cos(phi*pi/180.),\
                                 y + line_length * sin(phi*pi/180.)
                plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

            # plot cars
            for veh in self.all_vehicles:
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, 'black')

            # plot_interested vehs
            for mode, num in self.veh_mode_dict.items():
                for i in range(num):
                    veh = self.interested_vehs[mode][i]
                    veh_x = veh['x']
                    veh_y = veh['y']
                    veh_phi = veh['phi']
                    veh_l = veh['l']
                    veh_w = veh['w']
                    task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}

                    if is_in_plot_area(veh_x, veh_y):
                        plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                        task = MODE2TASK[mode]
                        color = task2color[task]
                        draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')

            # plot own car
            # dict(v_x=ego_dict['v_x'],
            #      v_y=ego_dict['v_y'],
            #      r=ego_dict['r'],
            #      x=ego_dict['x'],
            #      y=ego_dict['y'],
            #      phi=ego_dict['phi'],
            #      l=ego_dict['l'],
            #      w=ego_dict['w'],
            #      Corner_point=self.cal_corner_point_of_ego_car(ego_dict)
            #      alpha_f_bound=alpha_f_bound,
            #      alpha_r_bound=alpha_r_bound,
            #      r_bound=r_bound)

            ego_v_x = self.ego_dynamics['v_x']
            ego_v_y = self.ego_dynamics['v_y']
            ego_r = self.ego_dynamics['r']
            ego_x = self.ego_dynamics['x']
            ego_y = self.ego_dynamics['y']
            ego_phi = self.ego_dynamics['phi']
            ego_l = self.ego_dynamics['l']
            ego_w = self.ego_dynamics['w']
            ego_alpha_f = self.ego_dynamics['alpha_f']
            ego_alpha_r = self.ego_dynamics['alpha_r']
            alpha_f_bound = self.ego_dynamics['alpha_f_bound']
            alpha_r_bound = self.ego_dynamics['alpha_r_bound']
            r_bound = self.ego_dynamics['r_bound']

            plot_phi_line(ego_x, ego_y, ego_phi, 'red')
            draw_rotate_rec(ego_x, ego_y, ego_phi, ego_l, ego_w, 'red')

            # plot future data
            tracking_info = self.obs[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (self.num_future_data+1)]
            future_path = tracking_info[self.per_tracking_info_dim:]
            for i in range(self.num_future_data):
                delta_x, delta_y, delta_phi = future_path[i*self.per_tracking_info_dim:
                                                          (i+1)*self.per_tracking_info_dim]
                path_x, path_y, path_phi = ego_x+delta_x, ego_y+delta_y, ego_phi-delta_phi
                plt.plot(path_x, path_y, 'g.')
                plot_phi_line(path_x, path_y, path_phi, 'g')

            delta_, _, _ = tracking_info[:3]
            ax.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')
            indexs, points = self.ref_path.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y],np.float32))
            path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
            plt.plot(path_x, path_y, 'g.')
            delta_x, delta_y, delta_phi = ego_x - path_x, ego_y - path_y, ego_phi - path_phi

            # plot real time traj
            # try:
            #     color = ['b', 'lime']
            #     for i, item in enumerate(real_time_traj):
            #         if i == path_index:
            #             plt.plot(item.path[0], item.path[1], color=color[i], alpha=1.0)
            #         else:
            #             plt.plot(item.path[0], item.path[1], color=color[i], alpha=0.3)
            #         indexs, points = item.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
            #         path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
            #         plt.plot(path_x, path_y,  color=color[i])
            # except Exception:
            #     pass

            # for j, item_point in enumerate(self.real_path.feature_points_all):
            #     for k in range(len(item_point)):
            #         plt.scatter(item_point[k][0], item_point[k][1], c='g')

            # text
            text_x, text_y_start = -110, 60
            ge = iter(range(0, 1000, 4))
            plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
            plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
            plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
            plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
            plt.text(text_x, text_y_start - next(ge), 'delta_: {:.2f}m'.format(delta_))
            plt.text(text_x, text_y_start - next(ge), 'delta_x: {:.2f}m'.format(delta_x))
            plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
            plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
            plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))

            plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
            plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.exp_v))
            plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate bound: [{:.2f}, {:.2f}]'.format(-r_bound, r_bound))

            plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$: {:.2f} rad'.format(ego_alpha_f))
            plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$ bound: [{:.2f}, {:.2f}] '.format(-alpha_f_bound,
                                                                                                        alpha_f_bound))
            plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$: {:.2f} rad'.format(ego_alpha_r))
            plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$ bound: [{:.2f}, {:.2f}] '.format(-alpha_r_bound,
                                                                                                        alpha_r_bound))
            if self.action is not None:
                steer, a_x = self.action[0], self.action[1]
                plt.text(text_x, text_y_start - next(ge), r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
                plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

            text_x, text_y_start = 80, 60
            ge = iter(range(0, 1000, 4))

            # done info
            plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.done_type))

            # reward info
            if self.reward_info is not None:
                for key, val in self.reward_info.items():
                    plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

            # indicator for trajectory selection
            # text_x, text_y_start = -25, -65
            # ge = iter(range(0, 1000, 6))
            # if traj_return is not None:
            #     for i, value in enumerate(traj_return):
            #         if i==path_index:
            #             plt.text(text_x, text_y_start-next(ge), 'track_error={:.4f}, collision_risk={:.4f}'.format(value[0], value[1]), fontsize=14, color=color[i], fontstyle='italic')
            #         else:
            #             plt.text(text_x, text_y_start-next(ge), 'track_error={:.4f}, collision_risk={:.4f}'.format(value[0], value[1]), fontsize=12, color=color[i], fontstyle='italic')

            plt.show()
            plt.pause(0.001)

    def set_traj(self, trajectory):
        """set the real trajectory to reconstruct observation"""
        self.ref_path = trajectory


def test_end2end():
    env = CrossroadEnd2end(training_task='straight', num_future_data=0)
    obs = env.reset()
    i = 0
    done = 0
    while i < 100000:
        for j in range(80):
            # print(i)
            i += 1
            # action=2*np.random.random(2)-1
            if obs[4]<-18:
                action = np.array([0, 1], dtype=np.float32)
            else:
                action = np.array([0.5, 0.33], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            env.render()
        done = 0
        obs = env.reset()
        env.render()


if __name__ == '__main__':
    test_end2end()