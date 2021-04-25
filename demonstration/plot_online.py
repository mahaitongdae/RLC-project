import time
import numpy as np
from math import cos, sin, pi

import matplotlib.pyplot as plt

from utils.endtoend_env_utils import *
from utils.misc import TimerStat, find_closest_point


class Plot():
    def __init__(self, shared_list, lock, args):
        self.args = args
        self.shared_list = shared_list
        self.lock = lock
        self.task = self.args.task
        self.model_only_test = self.args.model_only_test
        self.step_old = -1
        self.acc_timer = TimerStat()
        left_construct_traj = np.load('./map/left_ref.npy')
        straight_construct_traj = np.load('./map/straight_ref.npy')
        right_construct_traj = np.load('./map/right_ref.npy')
        self.ref_path_all = {'left': left_construct_traj, 'straight': straight_construct_traj,
                             'right': right_construct_traj,
                             }
        self.ref_path = self.ref_path_all[self.task][0]  # todo

    def run(self):
        extension = 40
        light_line_width = 3
        dotted_line_style = '--'
        solid_line_style = '-'
        start_time = 0
        v_old = 0.
        plt.title("Crossroad")
        ax = plt.axes(xlim=(-CROSSROAD_HALF_WIDTH - extension, CROSSROAD_HALF_WIDTH + extension),
                      ylim=(-CROSSROAD_D_HEIGHT - extension, CROSSROAD_U_HEIGHT + extension))

        plt.axis("equal")
        plt.axis('off')

        def is_in_plot_area(x, y, tolerance=5):
            if -CROSSROAD_HALF_WIDTH - extension + tolerance < x < CROSSROAD_HALF_WIDTH + extension - tolerance and \
                    -CROSSROAD_D_HEIGHT - extension + tolerance < y < CROSSROAD_U_HEIGHT + extension - tolerance:
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
            x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                             y + line_length * sin(phi * pi / 180.)
            plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

        while True:
            plt.cla()
            plt.axis('off')
            ax.add_patch(plt.Rectangle((-CROSSROAD_HALF_WIDTH - extension, -CROSSROAD_D_HEIGHT - extension),
                                       2 * CROSSROAD_HALF_WIDTH + 2 * extension,
                                       CROSSROAD_D_HEIGHT + CROSSROAD_U_HEIGHT + 2 * extension, edgecolor='black',
                                       facecolor='none'))

            # ----------horizon--------------
            plt.plot([-CROSSROAD_HALF_WIDTH - extension, -CROSSROAD_HALF_WIDTH], [0, 0], color='black')
            plt.plot([CROSSROAD_HALF_WIDTH + extension, CROSSROAD_HALF_WIDTH], [0, 0], color='black')

            #
            for i in range(1, LANE_NUMBER_LR + 1):
                linestyle = dotted_line_style if i < LANE_NUMBER_LR else solid_line_style
                plt.plot([-CROSSROAD_HALF_WIDTH - extension, -CROSSROAD_HALF_WIDTH],
                         [i * LANE_WIDTH_LR, i * LANE_WIDTH_LR],
                         linestyle=linestyle, color='black')
                plt.plot([CROSSROAD_HALF_WIDTH + extension, CROSSROAD_HALF_WIDTH],
                         [i * LANE_WIDTH_LR, i * LANE_WIDTH_LR],
                         linestyle=linestyle, color='black')
                plt.plot([-CROSSROAD_HALF_WIDTH - extension, -CROSSROAD_HALF_WIDTH],
                         [-i * LANE_WIDTH_LR, -i * LANE_WIDTH_LR],
                         linestyle=linestyle, color='black')
                plt.plot([CROSSROAD_HALF_WIDTH + extension, CROSSROAD_HALF_WIDTH],
                         [-i * LANE_WIDTH_LR, -i * LANE_WIDTH_LR],
                         linestyle=linestyle, color='black')

            # ----------vertical----------------
            plt.plot([0, 0], [-CROSSROAD_D_HEIGHT - extension, -CROSSROAD_D_HEIGHT], color='black')
            plt.plot([0, 0], [CROSSROAD_U_HEIGHT + extension, CROSSROAD_U_HEIGHT], color='black')

            #
            for i in range(1, LANE_NUMBER_UD + 1):
                linestyle = dotted_line_style if i < LANE_NUMBER_UD else solid_line_style
                plt.plot([i * LANE_WIDTH_UD, i * LANE_WIDTH_UD], [-CROSSROAD_D_HEIGHT - extension, -CROSSROAD_D_HEIGHT],
                         linestyle=linestyle, color='black')
                plt.plot([i * LANE_WIDTH_UD, i * LANE_WIDTH_UD], [CROSSROAD_U_HEIGHT + extension, CROSSROAD_U_HEIGHT],
                         linestyle=linestyle, color='black')
                plt.plot([-i * LANE_WIDTH_UD, -i * LANE_WIDTH_UD],
                         [-CROSSROAD_D_HEIGHT - extension, -CROSSROAD_D_HEIGHT],
                         linestyle=linestyle, color='black')
                plt.plot([-i * LANE_WIDTH_UD, -i * LANE_WIDTH_UD], [CROSSROAD_U_HEIGHT + extension, CROSSROAD_U_HEIGHT],
                         linestyle=linestyle, color='black')

            # ----------Oblique--------------
            plt.plot([LANE_NUMBER_UD * LANE_WIDTH_UD, CROSSROAD_HALF_WIDTH],
                     [-CROSSROAD_D_HEIGHT, -LANE_NUMBER_LR * LANE_WIDTH_LR],
                     color='black')
            plt.plot([LANE_NUMBER_UD * LANE_WIDTH_UD, CROSSROAD_HALF_WIDTH],
                     [CROSSROAD_U_HEIGHT, LANE_NUMBER_LR * LANE_WIDTH_LR],
                     color='black')
            plt.plot([-LANE_NUMBER_UD * LANE_WIDTH_UD, -CROSSROAD_HALF_WIDTH],
                     [-CROSSROAD_D_HEIGHT, -LANE_NUMBER_LR * LANE_WIDTH_LR],
                     color='black')
            plt.plot([-LANE_NUMBER_UD * LANE_WIDTH_UD, -CROSSROAD_HALF_WIDTH],
                     [CROSSROAD_U_HEIGHT, LANE_NUMBER_LR * LANE_WIDTH_LR],
                     color='black')

            # ----------------------ref_path--------------------
            color = ['blue', 'coral', 'cyan', 'green']
            for index, path in enumerate(self.ref_path_all[self.task]):
                if index == self.shared_list[12]:
                    ax.plot(path[0], path[1], color=color[index], alpha=1.0)
                else:
                    ax.plot(path[0], path[1], color=color[index], alpha=0.3)


            # plot_ref = ['left', 'straight', 'right']  # 'left','straight','right', 'left_ref','straight_ref','right_ref'
            # for ref in plot_ref:
            #     ref_path = self.ref_path_all[ref][0]
            #     ax.plot(ref_path[0], ref_path[1])
            # ax.plot(self.ref_path[0], self.ref_path[1], color='g')  # todo:

            state_other = self.shared_list[4].copy()
            # plot cars
            for veh in state_other:
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = L
                veh_w = W
                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, 'black')

            interested_vehs = self.shared_list[10].copy()

            for i in range(int(len(interested_vehs) / 4)):  # TODO:
                veh_x = interested_vehs[4 * i + 0]
                veh_y = interested_vehs[4 * i + 1]
                # plt.text(veh_x, veh_y, i, fontsize=12)
                veh_phi = interested_vehs[4 * i + 3]
                veh_l = L
                veh_w = W
                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, 'b', linestyle='--')

            v_light = int(self.shared_list[13])  # todo
            if v_light == 0:
                v_color, h_color = 'black', 'black'  # 'green', 'red'
            elif v_light == 1:
                v_color, h_color = 'black', 'black'
            elif v_light == 2:
                v_color, h_color = 'black', 'black'
            else:
                v_color, h_color = 'black', 'black'

            plt.plot([(LANE_NUMBER_UD - 1) * LANE_WIDTH_UD, LANE_NUMBER_UD * LANE_WIDTH_UD],
                     [-CROSSROAD_D_HEIGHT, -CROSSROAD_D_HEIGHT],
                     color=v_color, linewidth=light_line_width)
            plt.plot([-LANE_NUMBER_UD * LANE_WIDTH_UD, -(LANE_NUMBER_UD - 1) * LANE_WIDTH_UD],
                     [CROSSROAD_U_HEIGHT, CROSSROAD_U_HEIGHT],
                     color=v_color, linewidth=light_line_width)
            plt.plot([0, (LANE_NUMBER_UD - 1) * LANE_WIDTH_UD], [-CROSSROAD_D_HEIGHT, -CROSSROAD_D_HEIGHT],
                     color=v_color, linewidth=light_line_width)
            plt.plot([0, -(LANE_NUMBER_UD - 1) * LANE_WIDTH_UD], [CROSSROAD_U_HEIGHT, CROSSROAD_U_HEIGHT],
                     color=v_color, linewidth=light_line_width)

            state_ego = self.shared_list[9].copy()
            ego_v = state_ego['VehicleSPeedAct']
            ego_steer = state_ego['SteerAngleAct']
            ego_gear = state_ego['AutoGear']
            ego_gps_v = state_ego['GpsSpeed']
            ego_north_v = state_ego['NorthVelocity']
            ego_east_v = state_ego['EastVelocity']
            ego_yaw_rate = state_ego['YawRate']
            ego_long_acc = state_ego['LongitudinalAcc']
            ego_lat_acc = state_ego['LateralAcc']
            ego_throttle = state_ego['Throttle']
            ego_brk = state_ego['BrkOn']
            ego_x = state_ego['GaussX']
            ego_y = state_ego['GaussY']
            ego_longitude = state_ego['Longitude']
            ego_latitude = state_ego['Latitude']
            ego_phi = state_ego['Heading']
            ego_l = L
            ego_w = W

            if not self.model_only_test:
                real_action_x = state_ego['model_x_in_real_action']
                real_action_y = state_ego['model_y_in_real_action']
                real_action_phi = state_ego['model_phi_in_real_action']
                plot_phi_line(real_action_x, real_action_y, real_action_phi, 'blue')
                draw_rotate_rec(real_action_x, real_action_y, real_action_phi, ego_l, ego_w, 'blue')

                model_action_x = state_ego['model_x_in_model_action']
                model_action_y = state_ego['model_y_in_model_action']
                model_action_phi = state_ego['model_phi_in_model_action']
                plot_phi_line(model_action_x, model_action_y, model_action_phi, 'coral')
                draw_rotate_rec(model_action_x, model_action_y, model_action_phi, ego_l, ego_w, 'coral')

            ego_l = L
            ego_w = W
            plot_phi_line(ego_x, ego_y, ego_phi, 'red')
            draw_rotate_rec(ego_x, ego_y, ego_phi, ego_l, ego_w, 'red')
            # model_x = state_ego['model_x']
            # model_y = state_ego['model_y']
            # model_phi = state_ego['model_phi']
            # draw_rotate_rec(model_x, model_y, model_phi, ego_l, ego_w, 'blue')

            time1 = time.time()
            delta_time = time1 - start_time
            acc_actual = (ego_v - v_old) / delta_time
            self.acc_timer.push(acc_actual)
            start_time = time.time()
            v_old = ego_v

            indexs, points = find_closest_point(self.ref_path, np.array([ego_x], np.float32),
                                                np.array([ego_y], np.float32))
            path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
            plt.plot(path_x, path_y, 'g.')
            delta_x, delta_y, delta_phi = ego_x - path_x, ego_y - path_y, ego_phi - path_phi

            # plot txt
            text_x, text_y_start = -110, 60
            ge = iter(range(0, 1000, 4))
            plt.text(text_x, text_y_start - next(ge), 'ego_GaussX: {:.2f}m'.format(ego_x))
            plt.text(text_x, text_y_start - next(ge), 'ego_GaussY: {:.2f}m'.format(ego_y))
            plt.text(text_x, text_y_start - next(ge), 'gps_v: {:.2f}m/s'.format(ego_gps_v))
            plt.text(text_x, text_y_start - next(ge), 'north_v: {:.2f}m/s'.format(ego_north_v))
            plt.text(text_x, text_y_start - next(ge), 'east_v: {:.2f}m/s'.format(ego_east_v))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_yaw_rate))
            plt.text(text_x, text_y_start - next(ge), r'longitude: ${:.2f}\degree$'.format(ego_longitude))
            plt.text(text_x, text_y_start - next(ge), r'latitude: ${:.2f}\degree$'.format(ego_latitude))
            plt.text(text_x, text_y_start - next(ge), r'long_acc: ${:.2f}m/s^2$'.format(ego_long_acc))
            plt.text(text_x, text_y_start - next(ge), r'lat_acc: ${:.2f}m/s^2$'.format(ego_lat_acc))

            plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
            plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
            plt.text(text_x, text_y_start - next(ge), 'delta_x: {:.2f}m'.format(delta_x))
            plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
            plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
            plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))

            decision = self.shared_list[8].copy()
            decision_steer = decision['SteerAngleAim']
            decision_torque = decision['Torque']
            decision_brkacc = decision['Deceleration']
            decision_Dec_flag = decision['Dec_flag']
            decision_Tor_flag = decision['Tor_flag']
            front_wheel_deg = decision['front_wheel_deg']
            acc = decision['a_x']

            text_x, text_y_start = 70, 60
            ge = iter(range(0, 1000, 4))

            # done info
            # plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.done_type))
            plt.text(text_x, text_y_start - next(ge), 'CAN')
            plt.text(text_x, text_y_start - next(ge), 'speed_act: {:.2f}m/s'.format(ego_v))
            plt.text(text_x, text_y_start - next(ge), r'steering_act: ${:.2f}\degree$'.format(ego_steer))
            plt.text(text_x, text_y_start - next(ge), 'throttle: {:.2f}'.format(ego_throttle))
            plt.text(text_x, text_y_start - next(ge), 'brake_on: {:.2f}'.format(ego_brk))
            plt.text(text_x, text_y_start - next(ge), 'gear: {:.2f}'.format(ego_gear))
            plt.text(text_x, text_y_start - next(ge), '  ')
            plt.text(text_x, text_y_start - next(ge), 'Decision')
            plt.text(text_x, text_y_start - next(ge), r'steer_aim_decision: ${:.2f}\degree$'.format(decision_steer))
            plt.text(text_x, text_y_start - next(ge), 'torque_decision: {:.2f}Nm'.format(decision_torque))
            plt.text(text_x, text_y_start - next(ge), 'torque_flag: {}'.format(decision_Tor_flag))
            plt.text(text_x, text_y_start - next(ge), r'brake_acc_decision: {:.2f}$m/s^2$'.format(decision_brkacc))
            plt.text(text_x, text_y_start - next(ge), 'deceleration_flag: {}'.format(decision_Dec_flag))
            plt.text(text_x, text_y_start - next(ge), '  ')
            plt.text(text_x, text_y_start - next(ge), 'Net out')
            plt.text(text_x, text_y_start - next(ge), 'front_wheel_deg: {:.2f}'.format(front_wheel_deg))
            plt.text(text_x, text_y_start - next(ge), r'acc: {:.2f}$m/s^2$'.format(acc))
            plt.text(text_x, text_y_start - next(ge), r'acc_actual: {:.2f}$m/s^2$'.format(self.acc_timer.mean))

            # plot text of trajectroy
            text_x, text_y_start = -40, -70
            ge = iter(range(0, 1000, 6))
            traj_return_value = self.shared_list[11]
            for i, value in enumerate(traj_return_value):
                if i == self.shared_list[12]:
                    plt.text(text_x, text_y_start - next(ge), 'Path reward={:.4f}, Collision risk={:.4f}'.format(value[0], value[1]),
                             fontsize=14, color=color[i], fontstyle='italic')
                else:
                    plt.text(text_x, text_y_start - next(ge), 'Path reward={:.4f}, Collision risk={:.4f}'.format(value[0], value[1]),
                             fontsize=10, color=color[i], fontstyle='italic')
            plt.pause(0.00001)



def static_plot():
    plot = Plot(None, 0, None, 'left')
    plot.run()
    plt.show()


if __name__ == '__main__':
    static_plot()
