import json
import time
from datetime import datetime
from math import pi

import bezier
import numpy as np
import tensorflow as tf
import zmq

from utils.endtoend_env_utils import *
from utils.load_policy import LoadPolicy
from utils.dynamics_and_models import *
from utils.coordi_convert import ROTATE_ANGLE

rotate4vy = ROTATE_ANGLE + 1.


def deal_with_phi_diff(phi_diff):
    phi_diff = tf.where(phi_diff > 180., phi_diff - 360., phi_diff)
    phi_diff = tf.where(phi_diff < -180., phi_diff + 360., phi_diff)
    return phi_diff


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-155495,  # front wheel cornering stiffness [N/rad]
                                   C_r=-155495,  # rear wheel cornering stiffness [N/rad]
                                   a=1.19,  # distance from CG to front axle [m]
                                   b=1.46,  # distance from CG to rear axle [m]
                                   mass=1520.,  # mass [kg]
                                   I_z=2642,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=0.8,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        # self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
        #                            C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
        #                            a=1.06,  # distance from CG to front axle [m]
        #                            b=1.85,  # distance from CG to rear axle [m]
        #                            mass=1412.,  # mass [kg]
        #                            I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
        #                            miu=1.0,  # tire-road friction coefficient
        #                            g=9.81,  # acceleration of gravity [m/s^2]
        #                            )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))
        self.states = np.array([[0., 0., 0., 1.75, -40., 90.]], dtype=np.float32)
        # self.states = np.array([[0., 0., 0., 1.75+3.5, -60., 90.]], dtype=np.float32)  # for right
        self.states = tf.convert_to_tensor(self.states, dtype=tf.float32)

    def f_xu(self, actions, tau):  # states and actions are tensors, [[], [], ...]
        v_x, v_y, r, x, y, phi = self.states[:, 0], self.states[:, 1], self.states[:, 2],\
                                 self.states[:, 3], self.states[:, 4], self.states[:, 5]
        phi = phi * np.pi / 180.
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
        C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
        a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
        b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
        mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
        I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
        v_x_next = v_x + tau * (a_x + v_y * r)
        v_x_next = tf.clip_by_value(v_x_next, 0.,10.)

        next_state = [v_x_next,
                      (mass * v_y * v_x + tau * (
                                  a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * tf.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                                  tau * (tf.square(a) * C_f + tf.square(b) * C_r) - I_z * v_x),
                      x + tau * (v_x * tf.cos(phi) - v_y * tf.sin(phi)),
                      y + tau * (v_x * tf.sin(phi) + v_y * tf.cos(phi)),
                      (phi + tau * r) * 180 / np.pi]

        return tf.stack(next_state, 1)

    def prediction(self, u_1, tau):
        self.states = self.f_xu(u_1, tau)
        return self.states.numpy()

    def replace_ith_state(self, i, var):
        states = self.states.numpy()
        final = []
        for idx, v in enumerate(states[0]):
            if idx == i:
                final.append(var)
            else:
                final.append(v)
        final = np.array(final, dtype=np.float32)[np.newaxis, :]
        self.states = tf.convert_to_tensor(final)

    def set_states(self, states):
        self.states = tf.convert_to_tensor(states, dtype=tf.float32)

    def get_states(self):
        return self.states.numpy()

    def model_step(self, state_gps, vehiclemode, action4model, delta_t, prefix):
        if vehiclemode == 0:
            v_x, r, x, y, phi = state_gps['GpsSpeed'], state_gps['YawRate'], state_gps['GaussX'], \
                                state_gps['GaussY'], state_gps['Heading']
            self.set_states(np.array([[v_x, 0., r, x, y, phi]]))
            out = [('model_vx_in_{}_action'.format(prefix), v_x),
                   ('model_vy_in_{}_action'.format(prefix), 0.),
                   ('model_r_in_{}_action'.format(prefix), r),
                   ('model_x_in_{}_action'.format(prefix), x),
                   ('model_y_in_{}_action'.format(prefix), y),
                   ('model_phi_in_{}_action'.format(prefix), phi),
                   ('model_front_wheel_rad_in_{}_action'.format(prefix), 0.),
                   ('model_acc_in_{}_action'.format(prefix), 0.),
                   ]
            return OrderedDict(out)
        else:
            front_wheel_rad, acc = action4model[0][0], action4model[0][1]
            states = self.states.numpy()
            v_x, v_y, r, x, y, phi = states[0][0], states[0][1], states[0][2], \
                                     states[0][3], states[0][4], states[0][5]
            out = [('model_vx_in_{}_action'.format(prefix), v_x),
                   ('model_vy_in_{}_action'.format(prefix), v_y),
                   ('model_r_in_{}_action'.format(prefix), r),
                   ('model_x_in_{}_action'.format(prefix), x),
                   ('model_y_in_{}_action'.format(prefix), y),
                   ('model_phi_in_{}_action'.format(prefix), phi),
                   ('model_front_wheel_rad_in_{}_action'.format(prefix), front_wheel_rad),
                   ('model_acc_in_{}_action'.format(prefix), acc),
                   ]
            self.prediction(action4model, delta_t)
            return OrderedDict(out)


class ReferencePath(object):
    def __init__(self, task, mode=None, ref_index=None):
        self.mode = mode
        self.exp_v = EXPECTED_V #TODO: temp
        self.task = task
        self.path_list = []
        self.path_len_list = []
        self._construct_ref_path(self.task)
        self.ref_index = np.random.choice(len(self.path_list)) if ref_index is None else ref_index
        self.path = self.path_list[self.ref_index]

    def set_path(self, path_index=None):
        self.ref_index = path_index
        self.path = self.path_list[self.ref_index]

    def _construct_ref_path(self, task):
        sl = 40  # straight length
        meter_pointnum_ratio = 30
        control_ext_x = 2 * CROSSROAD_HALF_WIDTH / 3. + 5.
        control_ext_y = 2 * CROSSROAD_D_HEIGHT / 3. + 3.
        if task == 'left':
            end_offsets = [LANE_WIDTH_LR*(i+0.5) for i in range(LANE_NUMBER_LR)] #TODO: temp
            start_offsets = [LANE_WIDTH_UD*0.5] #TODO: temp
            for start_offset in start_offsets:
                for i, end_offset in enumerate(end_offsets):
                    if i == 0:
                        end_offset += 0.2
                    control_point1 = start_offset, -CROSSROAD_D_HEIGHT
                    control_point2 = start_offset, -CROSSROAD_D_HEIGHT + control_ext_y
                    control_point3 = -CROSSROAD_HALF_WIDTH + control_ext_x, end_offset
                    control_point4 = -CROSSROAD_HALF_WIDTH, end_offset

                    node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                              [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                             dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(T * 2 * (control_ext_x + control_ext_y) / 4) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = LANE_WIDTH_UD/2 * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-CROSSROAD_D_HEIGHT - sl, -CROSSROAD_D_HEIGHT, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = np.linspace(-CROSSROAD_HALF_WIDTH, -CROSSROAD_HALF_WIDTH - sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)

                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1,
                                        xs_2 - xs_1) * 180 / pi
                    planed_trj = xs_1, ys_1, phis_1
                    self.path_list.append(planed_trj)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))

        elif task == 'straight':
            end_offsets = [LANE_WIDTH_UD*(i+0.5)-0.2 for i in range(LANE_NUMBER_UD)]  # todo ch
            start_offsets = [LANE_WIDTH_UD*0.5]
            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    control_point1 = start_offset, -CROSSROAD_D_HEIGHT
                    control_point2 = start_offset, -CROSSROAD_D_HEIGHT + control_ext_y
                    control_point3 = end_offset, CROSSROAD_U_HEIGHT - control_ext_y
                    control_point4 = end_offset, CROSSROAD_U_HEIGHT

                    node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                              [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]]
                                             , dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(CROSSROAD_U_HEIGHT + CROSSROAD_D_HEIGHT) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-CROSSROAD_D_HEIGHT - sl, -CROSSROAD_D_HEIGHT, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    end_straight_line_y = np.linspace(CROSSROAD_U_HEIGHT, CROSSROAD_U_HEIGHT + sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1,
                                        xs_2 - xs_1) * 180 / pi
                    planed_trj = xs_1, ys_1, phis_1
                    self.path_list.append(planed_trj)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))

        else:
            assert task == 'right'
            control_ext_y = 2 * CROSSROAD_D_HEIGHT / 5.+ 3.
            control_ext_x = 2 * CROSSROAD_HALF_WIDTH / 5.+ 3.
            end_offsets = [-LANE_WIDTH_LR * (i + 0.5) for i in range(LANE_NUMBER_LR)]
            start_offsets = [LANE_WIDTH_UD*(LANE_NUMBER_UD-0.5)]

            for start_offset in start_offsets:
                for i, end_offset in enumerate(end_offsets):
                    if i == 0:
                        end_offset -= 0.2
                    control_point1 = start_offset, -CROSSROAD_D_HEIGHT
                    control_point2 = start_offset, -CROSSROAD_D_HEIGHT + control_ext_y
                    control_point3 = CROSSROAD_HALF_WIDTH - control_ext_x, end_offset
                    control_point4 = CROSSROAD_HALF_WIDTH, end_offset

                    node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                              [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                             dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(pi/2*(CROSSROAD_D_HEIGHT-LANE_WIDTH_LR*(LANE_NUMBER_LR/2-0.5))) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-CROSSROAD_D_HEIGHT - sl, -CROSSROAD_D_HEIGHT, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = np.linspace(CROSSROAD_HALF_WIDTH, CROSSROAD_HALF_WIDTH + sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1,
                                        xs_2 - xs_1) * 180 / pi
                    planed_trj = xs_1, ys_1, phis_1
                    self.path_list.append(planed_trj)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))

    def find_closest_point(self, xs, ys, ratio=10):
        path_len = len(self.path[0])
        reduced_idx = np.arange(0, path_len, ratio)
        reduced_len = len(reduced_idx)
        reduced_path_x, reduced_path_y = self.path[0][reduced_idx], self.path[1][reduced_idx]
        xs_tile = tf.tile(tf.reshape(xs, (-1, 1)), tf.constant([1, reduced_len]))
        ys_tile = tf.tile(tf.reshape(ys, (-1, 1)), tf.constant([1, reduced_len]))
        pathx_tile = tf.tile(tf.reshape(reduced_path_x, (1, -1)), tf.constant([len(xs), 1]))
        pathy_tile = tf.tile(tf.reshape(reduced_path_y, (1, -1)), tf.constant([len(xs), 1]))

        dist_array = tf.square(xs_tile - pathx_tile) + tf.square(ys_tile - pathy_tile)

        indexs = tf.argmin(dist_array, 1) * ratio
        return indexs, self.indexs2points(indexs)

    def future_n_data(self, current_indexs, n):
        future_data_list = []
        current_indexs = tf.cast(current_indexs, tf.int32)
        for _ in range(n):
            current_indexs += 80
            current_indexs = tf.where(current_indexs >= len(self.path[0]) - 2, len(self.path[0]) - 2, current_indexs)
            future_data_list.append(self.indexs2points(current_indexs))
        return future_data_list

    def indexs2points(self, indexs):
        indexs = tf.where(indexs >= 0, indexs, 0)
        indexs = tf.where(indexs < len(self.path[0]), indexs, len(self.path[0])-1)
        points = tf.gather(self.path[0], indexs), \
                 tf.gather(self.path[1], indexs), \
                 tf.gather(self.path[2], indexs)

        return points[0], points[1], points[2]

    def tracking_error_vector(self, ego_xs, ego_ys, ego_phis, ego_vs, n):
        def two2one(ref_xs, ref_ys):
            if self.task == 'left':
                delta_ = tf.sqrt(tf.square(ego_xs - (-CROSSROAD_HALF_WIDTH)) + tf.square(ego_ys - (-CROSSROAD_D_HEIGHT))) - \
                         tf.sqrt(tf.square(ref_xs - (-CROSSROAD_HALF_WIDTH)) + tf.square(ref_ys - (-CROSSROAD_D_HEIGHT)))
                delta_ = tf.where(ego_ys < -CROSSROAD_D_HEIGHT, ego_xs - ref_xs, delta_)
                delta_ = tf.where(ego_xs < -CROSSROAD_HALF_WIDTH, ego_ys - ref_ys, delta_)
                return -delta_
            elif self.task == 'straight':
                delta_ = ego_xs - ref_xs
                return -delta_
            else:
                assert self.task == 'right'
                delta_ = -(tf.sqrt(tf.square(ego_xs - CROSSROAD_HALF_WIDTH) + tf.square(ego_ys - (-CROSSROAD_D_HEIGHT))) -
                           tf.sqrt(tf.square(ref_xs - CROSSROAD_HALF_WIDTH) + tf.square(ref_ys - (-CROSSROAD_D_HEIGHT))))
                delta_ = tf.where(ego_ys < -CROSSROAD_D_HEIGHT, ego_xs - ref_xs, delta_)
                delta_ = tf.where(ego_xs > CROSSROAD_HALF_WIDTH, -(ego_ys - ref_ys), delta_)
                return -delta_

        indexs, current_points = self.find_closest_point(ego_xs, ego_ys)
        # print('Index:', indexs.numpy(), 'points:', current_points[:])
        n_future_data = self.future_n_data(indexs, n)

        tracking_error = tf.stack([two2one(current_points[0], current_points[1]),
                                           deal_with_phi_diff(ego_phis - current_points[2]),
                                           ego_vs - self.exp_v], 1)

        final = tracking_error
        if n > 0:
            future_points = tf.concat([tf.stack([ref_point[0] - ego_xs,
                                                 ref_point[1] - ego_ys,
                                                 deal_with_phi_diff(ego_phis - ref_point[2])], 1)
                                       for ref_point in n_future_data], 1)
            final = tf.concat([final, future_points], 1)

        return final


class Controller(object):
    def __init__(self, shared_list, receive_index, lock, args):
        self.args = args
        self.time_out = 0
        self.task = self.args.task
        self.true_ss = self.args.true_ss
        self.ss_con_v = self.args.ss_con_v
        self.ref_path = ReferencePath(self.task)
        self.num_future_data = 0
        self.noise_factor = self.args.noise_factor
        self.model = LoadPolicy(self.args.load_dir, self.args.load_ite)
        self.steer_factor = 15
        self.shared_list = shared_list
        self.read_index_old = 0
        self.receive_index_shared = receive_index
        self.model_only_test = self.args.model_only_test
        self.clipped_v = self.args.clipped_v
        # self.read_index_old = Info_List[0]

        self.lock = lock
        context = zmq.Context()
        self.socket_pub = context.socket(zmq.PUB)
        self.socket_pub.bind("tcp://*:6970")
        self.time_initial = time.time()
        self.step = 0
        self.if_save = self.args.if_save
        self.save_path = self.args.result_dir
        self.t_interval = 0
        self.time_decision = 0
        self.time_in = time.time()

        self.model_driven_by_model_action = VehicleDynamics()
        self.model_driven_by_real_action = VehicleDynamics()

        self.predict_model = EnvironmentModel(self.task)
        self.run_time = 0.

    def _construct_ego_vector(self, state_gps, model_flag):
        if model_flag:
            v_x, v_y, r, x, y, phi = state_gps['v_x'], state_gps['v_y'], state_gps['r'],\
                                     state_gps['x'],  state_gps['y'], state_gps['phi']
            self.ego_info_dim = 6
            ego_feature = [v_x, v_y, r, x, y, phi]
            return np.array(ego_feature, dtype=np.float32)
        else:
            ego_phi = state_gps['Heading']
            ego_phi_rad = ego_phi * np.pi / 180.
            ego_x, ego_y = state_gps['GaussX'], state_gps['GaussY']
            v_in_y_coord = state_gps['NorthVelocity']*np.cos(rotate4vy*np.pi/180) - state_gps['EastVelocity']*np.sin(rotate4vy*np.pi/180)
            v_in_x_coord = state_gps['NorthVelocity']*np.sin(rotate4vy*np.pi/180) + state_gps['EastVelocity']*np.cos(rotate4vy*np.pi/180)
            ego_v_x = v_in_y_coord * np.sin(ego_phi_rad) + v_in_x_coord * np.cos(ego_phi_rad)
            ego_v_y = v_in_y_coord * np.cos(ego_phi_rad) - v_in_x_coord * np.sin(ego_phi_rad)  # todo: check the sign
            ego_v_y = - ego_v_y
            ego_r = state_gps['YawRate']                      # rad/s
            self.ego_info_dim = 6
            ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi]
            return np.array(ego_feature, dtype=np.float32)

    def _construct_veh_vector(self, ego_x, ego_y, state_others):
        all_vehicles = state_others
        v_light = int(self.shared_list[13])
        vehs_vector = []
        name_setting = dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i')

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
            if self.task != 'right':
                if v_light != 0 and ego_y < -CROSSROAD_D_HEIGHT:
                    dl.append(dict(x=LANE_WIDTH_UD/2, y=-CROSSROAD_D_HEIGHT + 2.5, v=0., phi=90, l=5, w=2.5, route=None))
                    du.append(dict(x=LANE_WIDTH_UD/2, y=-CROSSROAD_D_HEIGHT + 2.5, v=0., phi=90, l=5, w=2.5, route=None))
            # todo: whether add dr for left and straight; right task has no virtual front car

            # fetch veh in range
            if task == 'left':
                dl = list(filter(lambda v: v['x'] > -CROSSROAD_HALF_WIDTH-10 and v['y'] > ego_y-2, dl))
                du = list(filter(lambda v: ego_y-2 < v['y'] < CROSSROAD_U_HEIGHT+10 and v['x'] < ego_x+2, du))
                dr = list(filter(lambda v: v['x'] < ego_x+2 and v['y'] > ego_y-2, dr))
            elif task == 'straight':
                dl = list(filter(lambda v: v['x'] > ego_x-2 and v['y'] > ego_y - 2, dl))
                du = list(filter(lambda v: ego_y - 2 < v['y'] < CROSSROAD_U_HEIGHT + 10, du))
                dr = list(filter(lambda v: v['x'] < ego_x+2 and v['y'] > ego_y-2, dr))
            else:
                assert task == 'right'
                dl = list(filter(lambda v: v['x'] > ego_x - 2 and v['y'] > ego_y - 2, dl))
                du = list(filter(lambda v: v['x'] > ego_x - 2 and v['y'] > ego_y - 2, du))
                dr = list(filter(lambda v: v['x'] < CROSSROAD_HALF_WIDTH + 10 and v['y'] > ego_y-2, dr))

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(lambda v: v['x'] < CROSSROAD_HALF_WIDTH + 10 and v['y'] < CROSSROAD_U_HEIGHT + 10, ru))

            if task == 'straight':
                ur = list(filter(lambda v: v['x'] < ego_x + 2 and ego_y < v['y'] < CROSSROAD_U_HEIGHT + 10, ur))
            elif task == 'right':
                ur = list(filter(lambda v: v['x'] < CROSSROAD_HALF_WIDTH+10 and v['y'] < CROSSROAD_U_HEIGHT, ur))
            ud = list(filter(lambda v: max(ego_y-2, -CROSSROAD_D_HEIGHT) < v['y'] < CROSSROAD_U_HEIGHT
                                       and ego_x > v['x'] and ego_y > -CROSSROAD_D_HEIGHT, ud))
            ul = list(filter(lambda v: -CROSSROAD_HALF_WIDTH-10 < v['x'] < ego_x and v['y'] < CROSSROAD_U_HEIGHT, ul))

            lu = lu  # not interest in case of traffic light
            lr = list(filter(lambda v: -CROSSROAD_HALF_WIDTH-10 < v['x'] < CROSSROAD_HALF_WIDTH+10, lr))  # interest of right
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
            mode2fillvalue = dict(dl=dict(x=LANE_WIDTH_UD/2, y=-(CROSSROAD_D_HEIGHT+30), v=0, phi=90, w=2.5, l=5, route=('1o', '4i')),
                                  du=dict(x=LANE_WIDTH_UD/2, y=-(CROSSROAD_D_HEIGHT+30), v=0, phi=90, w=2.5, l=5, route=('1o', '3i')),
                                  dr=dict(x=LANE_WIDTH_UD*(LANE_NUMBER_UD-0.5), y=-(CROSSROAD_D_HEIGHT+30), v=0, phi=90, w=2.5, l=5, route=('1o', '2i')),
                                  ru=dict(x=(CROSSROAD_HALF_WIDTH+15), y=LANE_WIDTH_LR*(LANE_NUMBER_LR-0.5), v=0, phi=180, w=2.5, l=5, route=('2o', '3i')),
                                  ur=dict(x=-LANE_WIDTH_UD/2, y=(CROSSROAD_U_HEIGHT+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '2i')),
                                  ud=dict(x=-LANE_WIDTH_UD*0.5, y=(CROSSROAD_U_HEIGHT+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '1i')),
                                  ul=dict(x=-LANE_WIDTH_UD*(LANE_NUMBER_UD-0.5), y=(CROSSROAD_U_HEIGHT+20), v=0, phi=-90, w=2.5, l=5, route=('3o', '4i')),
                                  lr=dict(x=-(CROSSROAD_HALF_WIDTH+20), y=-LANE_WIDTH_LR*1.5, v=0, phi=0, w=2.5, l=5, route=('4o', '2i')))

            tmp = OrderedDict()
            for mode, num in VEHICLE_MODE_DICT[task].items():
                tmp[mode] = slice_or_fill(eval(mode), mode2fillvalue[mode], num)

            return tmp

        list_of_interested_veh_dict = []
        self.interested_vehs = filter_interested_vehicles(all_vehicles, self.task)
        for part in list(self.interested_vehs.values()):
            list_of_interested_veh_dict.extend(part)
        self.per_veh_info_dim = 4
        for veh in list_of_interested_veh_dict:
            veh_x, veh_y, veh_v, veh_phi = veh['x'], veh['y'], veh['v'], veh['phi']
            vehs_vector.extend([veh_x, veh_y, veh_v, veh_phi])
        return np.array(vehs_vector, dtype=np.float32)

    def _get_obs(self, state_gps, state_others, model_flag):
        if model_flag:
            ego_v_x, _, _, ego_x, ego_y, ego_phi = state_gps['v_x'], state_gps['v_y'], state_gps['r'],\
                                                   state_gps['x'],  state_gps['y'], state_gps['phi']
        else:
            ego_x, ego_y = state_gps['GaussX'], state_gps['GaussY']
            ego_phi = state_gps['Heading']
            ego_v_x, ego_v_y = state_gps['GpsSpeed'], 0.
        vehs_vector = self._construct_veh_vector(ego_x, ego_y, state_others)
        ego_vector = self._construct_ego_vector(state_gps, model_flag)
        tracking_error = self.ref_path.tracking_error_vector(np.array([ego_x], dtype=np.float32),
                                                             np.array([ego_y], dtype=np.float32),
                                                             np.array([ego_phi], dtype=np.float32),
                                                             np.array([ego_v_x], dtype=np.float32),
                                                             self.num_future_data).numpy()[0]
        self.per_tracking_info_dim = 3
        vector = np.concatenate((ego_vector, tracking_error, vehs_vector), axis=0)
        veh_idx_start = self.ego_info_dim + self.per_tracking_info_dim * (self.num_future_data + 1)

        noise = np.zeros_like(vector)
        nf = self.noise_factor
        noise[self.ego_info_dim] = nf * np.clip(np.random.normal(0, 0.017), -0.051, 0.051)
        noise[self.ego_info_dim+1] = nf * np.clip(np.random.normal(0, 0.17), -0.51, 0.51)
        for veh_idx in range(int(len(vehs_vector)/self.per_veh_info_dim)):
            noise[veh_idx_start+self.per_veh_info_dim*veh_idx] = nf * np.clip(np.random.normal(0, 0.05), -0.15, 0.15)
            noise[veh_idx_start+self.per_veh_info_dim*veh_idx+1] = nf * np.clip(np.random.normal(0, 0.05), -0.15, 0.15)
            noise[veh_idx_start+self.per_veh_info_dim*veh_idx+2] = nf * np.clip(np.random.normal(0, 0.05), -0.15, 0.15)
            noise[veh_idx_start+self.per_veh_info_dim*veh_idx+3] = nf * np.clip(np.random.normal(0, 1.4), -5.2, 5.2)

        vector_with_noise = vector + noise
        obs_dict = OrderedDict(ego_vx=ego_vector[0], ego_vy=ego_vector[1], ego_r=ego_vector[2], ego_x=ego_vector[3],
                               ego_y=ego_vector[4], ego_phi=ego_vector[5],
                               tracking_delta_y=tracking_error[0], tracking_delta_phi=tracking_error[1],
                               tracking_delta_v=tracking_error[2],
                               )
        for i in range(int(len(vehs_vector)/self.per_veh_info_dim)):
            obs_dict.update({'other{}_x'.format(i): vehs_vector[self.per_veh_info_dim*i],
                             'other{}_y'.format(i): vehs_vector[self.per_veh_info_dim*i+1],
                             'other{}_v'.format(i): vehs_vector[self.per_veh_info_dim*i+2],
                             'other{}_phi'.format(i): vehs_vector[self.per_veh_info_dim*i+3]})
        return vector_with_noise, obs_dict, vehs_vector  # todo: if output vector without noise

    def _action_transformation_for_end2end(self, action, obs_dict, path_index=None):  # [-1, 1]
        ego_v_x = obs_dict['ego_vx']
        ego_x = obs_dict['ego_x']
        ego_y = obs_dict['ego_y']
        delta_y = obs_dict['tracking_delta_y']
        delta_phi = obs_dict['tracking_delta_phi']
        delta_v = obs_dict['tracking_delta_v']

        torque_clip = 100. if ego_v_x > self.clipped_v else 250.         # todo: clipped v
        action = np.clip(action, -1.0, 1.0)
        front_wheel_norm_rad, a_x_norm = action[0], action[1]
        front_wheel_deg = 0.4 / pi * 180 * front_wheel_norm_rad
        steering_wheel = front_wheel_deg * self.steer_factor

        # rules for path selection for right and left task
        if self.task == 'left' and path_index == 0:
            steering_wheel = -50 * delta_y - 10 * delta_phi
        if self.task == 'right' and path_index == 3:
            steering_wheel = -50 * delta_y - 10 * delta_phi

        # rules on straight line for all tasks
        if (ego_y < -CROSSROAD_D_HEIGHT) or (self.task == 'straight' and ego_y > CROSSROAD_U_HEIGHT) \
                                         or (self.task == 'right' and ego_x > CROSSROAD_HALF_WIDTH) \
                                         or (self.task == 'left' and ego_x < -CROSSROAD_HALF_WIDTH):
            if abs(delta_y) < 0.5 and abs(delta_phi) < 5.:
                steering_wheel = np.clip(steering_wheel, -10., 10)

        steering_wheel = np.clip(steering_wheel, -360., 360)
        a_x = 2.25*a_x_norm - 0.75
        if a_x > -0.1:
            # torque = np.clip(a_x * 300., 0., 350.)
            torque = np.clip((a_x+0.1-0.4)/0.4*80+150., 0., torque_clip)
            decel = 0.
            tor_flag = 1
            dec_flag = 0
        else:
            torque = 0.
            # decel = np.clip(-a_x, 0., 4.)
            decel = a_x-0.1
            tor_flag = 0
            dec_flag = 1

        # out: steer_wheel_deg, torque, deceleration, tor_flag, dec_flag:
        # [-360,360]deg, [0., 350,]N (1), [0, 5]m/s^2 (0.05)
        return steering_wheel, torque, decel, tor_flag, dec_flag, front_wheel_deg, a_x

    @tf.function
    def _safety_sheild(self, obs, action, con_v):
        flag = 0
        if self.true_ss is None:
            safe_action = action[0]
        elif self.true_ss == 'pred':
            self.predict_model.add_traj(obs, self.ref_path, mode='selecting')
            _, veh2veh4real = self.predict_model.safety_calculation(obs,action)
            veh2veh4real = veh2veh4real[0]
            if veh2veh4real != 0:
                flag = 1
                tf.print('original action will cause collision!!!')
                safe_action = tf.convert_to_tensor([0., -1.0], dtype=tf.float32)
            else:
                safe_action = action[0]
        else:
            assert self.true_ss == 'con_v'
            if con_v > self.ss_con_v:
                flag = 1
                safe_action = tf.convert_to_tensor([0., -1.0], dtype=tf.float32)
            else:
                safe_action = action[0]
        return safe_action, flag

    def hier_decision(self, state_gps, state_other, model_flag):
        traj_num = LANE_NUMBER_LR if self.task == 'right' or self.task == 'left' else LANE_NUMBER_UD
        obs_list = []
        for traj_index in range(traj_num):
            self.ref_path.set_path(traj_index)
            obs, _, _ = self._get_obs(state_gps, state_other, model_flag=model_flag)
            obs_list.append(obs)
        all_obs = tf.convert_to_tensor(obs_list, dtype=tf.float32)
        obj_vs, con_vs = self.model.values(all_obs)
        traj_return_value = np.stack([obj_vs.numpy(), con_vs.numpy()], axis=1)
        path_selection = 0
        path_index = np.argmax(traj_return_value[:, path_selection])
        self.ref_path.set_path(path_index)
        obs, obs_dict, veh_vec = self._get_obs(state_gps, state_other, model_flag=model_flag)
        action = self.model.run(obs)
        obs, action = tf.convert_to_tensor(obs[np.newaxis, :]), tf.convert_to_tensor(action[np.newaxis, :])
        action, ss_flag = self._safety_sheild(obs, action, con_vs[path_index])

        path_dict = OrderedDict({'obj_value': traj_return_value[:, 0].tolist(),
                                 'con_value': traj_return_value[:, 1].tolist(),
                                 'index': [path_index, path_selection],
                                 'ss_flag': [ss_flag.numpy()]
                                 })
        return path_index, traj_return_value, action.numpy(), obs_dict, veh_vec, path_dict

    def run(self):
        start_time = time.time()
        with open(self.save_path + '/record.txt', 'a') as file_handle:
            file_handle.write(str("保存时间：" + datetime.now().strftime("%Y%m%d_%H%M%S")))
            file_handle.write('\n')
            while True:
                time.sleep(0.07)
                if self.model_only_test:   # test using model
                    model_state = self.model_driven_by_model_action.get_states()[0]
                    v_x, v_y, r, x, y, phi = model_state[0], model_state[1], model_state[2],\
                                             model_state[3], model_state[4], model_state[5]
                    with self.lock:
                        state_gps = self.shared_list[0].copy()
                        time_receive_gps = self.shared_list[1]
                        state_can = self.shared_list[2].copy()
                        time_receive_can = self.shared_list[3]
                        state_other = self.shared_list[4].copy()
                        time_receive_radar = 0.

                    state_ego = OrderedDict()
                    state_gps.update(dict(GaussX=x, GaussY=y, Heading=phi, GpsSpeed=v_x))  # only for plot online
                    state_ego.update(state_gps)
                    state_ego.update(state_can)

                    state_gps_modified_by_model = dict(v_x=v_x, v_y=v_y, r=r, x=x, y=y, phi=phi)
                    self.time_in = time.time()
                    path_index, traj_return_value, action, obs_dict, veh_vec, path_dict = \
                        self.hier_decision(state_gps_modified_by_model, state_other, model_flag=True)

                    steer_wheel_deg, torque, decel, tor_flag, dec_flag, front_wheel_deg, a_x = \
                        self._action_transformation_for_end2end(action, obs_dict, path_index)
                    action = np.array([[front_wheel_deg * np.pi / 180, a_x]], dtype=np.float32)
                    state_model_in_model_action = self.model_driven_by_model_action.model_step(state_gps, 1,
                                                                                               action,
                                                                                               time.time() - start_time,
                                                                                               'model')
                    state_ego.update(state_model_in_model_action)
                    start_time = time.time()
                    control = {'Decision': {
                        'Control': {#'VehicleSpeedAim': 20/3.6,
                                    'Deceleration': decel,
                                    'Torque': torque,
                                    'Dec_flag': dec_flag,
                                    'Tor_flag': tor_flag,
                                    'SteerAngleAim': np.float64(steer_wheel_deg+1.7),
                                    'VehicleGearAim': 1,
                                    'IsValid': True}}}
                    json_cotrol = json.dumps(control)
                    self.socket_pub.send(json_cotrol.encode('utf-8'))

                    time_decision = time.time() - self.time_in
                    self.run_time = time.time() - self.time_initial

                    decision = OrderedDict({'Deceleration': decel,  # [m/s^2]
                                            'Torque': torque,  # [N*m]
                                            'Dec_flag': dec_flag,
                                            'Tor_flag': tor_flag,
                                            'SteerAngleAim': steer_wheel_deg,  # [deg]
                                            'front_wheel_deg': front_wheel_deg,
                                            'a_x': a_x})  # [m/s^2]

                    with self.lock:
                        self.shared_list[6] = self.step
                        self.shared_list[7] = self.run_time
                        self.shared_list[8] = decision.copy()
                        self.shared_list[9] = state_ego.copy()
                        self.shared_list[10] = list(veh_vec)
                        self.shared_list[11] = traj_return_value
                        self.shared_list[12] = path_index
                        self.shared_list[14] = path_dict['ss_flag'][0]

                    self.step += 1
                else:  # real test
                    shared_index = self.receive_index_shared.value
                    if shared_index > self.read_index_old:
                        self.read_index_old = shared_index
                        with self.lock:
                            state_gps = self.shared_list[0].copy()
                            time_receive_gps = self.shared_list[1]
                            state_can = self.shared_list[2].copy()
                            time_receive_can = self.shared_list[3]
                            state_other = self.shared_list[4].copy()
                            time_receive_radar = 0.

                        state_ego = OrderedDict()
                        state_ego.update(state_gps)
                        state_ego.update(state_can)

                        self.time_in = time.time()
                        obs, obs_dict, veh_vec = self._get_obs(state_gps, state_other, model_flag=False)
                        action = self.model.run(obs)
                        steer_wheel_deg, torque, decel, tor_flag, dec_flag, front_wheel_deg, a_x = \
                            self._action_transformation_for_end2end(action, obs_dict)
                        # ==============================================================================================
                        # ------------------drive model in real action---------------------------------
                        realaction4model = np.array([[front_wheel_deg*np.pi/180, a_x]], dtype=np.float32)
                        state_model_in_real_action = self.model_driven_by_real_action.model_step(state_gps, state_can['VehicleMode'],
                                                                                  realaction4model, time.time()-start_time, 'real')
                        state_ego.update(state_model_in_real_action)
                        # ------------------drive model in real action---------------------------------

                        # ------------------drive model in model action---------------------------------
                        if self.step % 5 == 0:
                            v_in_y_coord = state_gps['NorthVelocity'] * np.cos(rotate4vy * np.pi / 180) - state_gps[
                                'EastVelocity'] * np.sin(rotate4vy * np.pi / 180)
                            v_in_x_coord = state_gps['NorthVelocity'] * np.sin(rotate4vy * np.pi / 180) + state_gps[
                                'EastVelocity'] * np.cos(rotate4vy * np.pi / 180)
                            ego_phi = state_gps['Heading']
                            ego_phi_rad = ego_phi * np.pi / 180.
                            ego_vx = v_in_y_coord * np.sin(ego_phi_rad) + v_in_x_coord * np.cos(ego_phi_rad)
                            ego_vy = v_in_y_coord * np.cos(ego_phi_rad) - v_in_x_coord * np.sin(ego_phi_rad)
                            ego_vy = -ego_vy
                            ego_r, ego_x, ego_y, ego_phi = state_gps['YawRate'], state_gps['GaussX'], state_gps[
                                'GaussY'], state_gps['Heading']
                            self.model_driven_by_model_action.set_states(np.array([[ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi]], dtype=np.float32))
                        state_driven_by_model_action = self.model_driven_by_model_action.get_states()[0]
                        v_x, v_y, r, x, y, phi = state_driven_by_model_action[0], state_driven_by_model_action[1], state_driven_by_model_action[2], \
                                                 state_driven_by_model_action[3], state_driven_by_model_action[4], state_driven_by_model_action[5]
                        state_gps_modified_by_model = dict(v_x=v_x, v_y=v_y, r=r, x=x, y=y, phi=phi)
                        path_index, traj_return_value, action_model, obs_dict_model, veh_vec_model, path_dict = \
                            self.hier_decision(state_gps_modified_by_model, state_other, model_flag=True)
                        steer_wheel_deg_model, torque_model, decel_model, tor_flag_model, dec_flag_model, front_wheel_deg_model, a_x_model = \
                            self._action_transformation_for_end2end(action_model, obs_dict_model, path_index)
                        modelaction4model = np.array([[front_wheel_deg_model*np.pi/180, a_x_model]], dtype=np.float32)
                        state_model_in_model_action = self.model_driven_by_model_action.model_step(state_gps, state_can['VehicleMode'],
                                                                                                   modelaction4model,
                                                                                                   time.time() - start_time,
                                                                                                   'model')
                        state_ego.update(state_model_in_model_action)
                        # ------------------drive model in model action---------------------------------
                        # ==============================================================================================

                        start_time = time.time()
                        control = {'Decision': {
                            'Control': {  # 'VehicleSpeedAim': 20/3.6,
                                'Deceleration': decel_model,
                                'Torque': torque_model,
                                'Dec_flag': dec_flag_model,
                                'Tor_flag': tor_flag_model,
                                'SteerAngleAim': np.float64(steer_wheel_deg_model + 1.7),
                                'VehicleGearAim': 1,
                                'IsValid': True}}}
                        json_cotrol = json.dumps(control)
                        self.socket_pub.send(json_cotrol.encode('utf-8'))
                        time_decision = time.time() - self.time_in
                        self.run_time = time.time() - self.time_initial

                        decision = OrderedDict({'Deceleration': decel_model,  # [m/s^2]
                                                'Torque': torque_model,  # [N*m]
                                                'Dec_flag': dec_flag_model,
                                                'Tor_flag': tor_flag_model,
                                                'SteerAngleAim': steer_wheel_deg_model,  # [deg]
                                                'front_wheel_deg': front_wheel_deg_model,
                                                'a_x': a_x_model})  # [m/s^2]

                        with self.lock:
                            self.shared_list[6] = self.step
                            self.shared_list[7] = self.run_time
                            self.shared_list[8] = decision.copy()
                            self.shared_list[9] = state_ego.copy()
                            self.shared_list[10] = list(veh_vec)
                            self.shared_list[11] = traj_return_value
                            self.shared_list[12] = path_index  # 13 is v light
                            self.shared_list[14] = path_dict['ss_flag'][0]

                        self.step += 1

                if self.if_save:
                    if decision != {} and state_ego != {} and state_other != {}:
                        file_handle.write("Decision ")
                        for k1, v1 in decision.items():
                            file_handle.write(k1 + ":" + str(v1) + ", ")
                        file_handle.write('\n')

                        file_handle.write("State_ego ")
                        for k2, v2 in state_ego.items():
                            file_handle.write(k2 + ":" + str(v2) + ", ")
                        file_handle.write('\n')

                        file_handle.write("State_other ")
                        file_handle.write('\n')

                        file_handle.write("Obs_dict ")
                        for k4, v4 in obs_dict.items():
                            file_handle.write(k4 + ":" + str(v4) + ", ")
                        file_handle.write('\n')

                        file_handle.write("Path ")
                        for k4, v4 in path_dict.items():
                            file_handle.write(k4 + ":" + str(v4) + "| ")
                        file_handle.write('\n')

                        file_handle.write("Time Time:" + str(self.run_time) + ", " +
                                          "time_decision:"+str(time_decision) + ", " +
                                          "time_receive_gps:"+str(time_receive_gps) + ", " +
                                          "time_receive_can:"+str(time_receive_can) + ", " +
                                          "time_receive_radar:"+str(time_receive_radar)+ ", " + '\n')


if __name__ == "__main__":
    from main import built_parser
    import multiprocessing as mp
    args = built_parser()
    os.makedirs(args.result_dir)
    shared_list = mp.Manager().list([0.] * 11)
    receive_index = mp.Value('d', 0.0)
    lock = mp.Lock()
    publisher = Controller(shared_list, receive_index, args.if_save, lock,
                                                    args.task, args.noise_factor, args.load_dir,
                                                    args.load_ite, args.result_dir, args.model_only_test,
                                                    args.clipped_v)
    ref_path = ReferencePath('left')
    for i in range(4):
        ref_path.set_path(i)
        print(ref_path.ref_index)
        path = ref_path.path
        print(ref_path.path)
