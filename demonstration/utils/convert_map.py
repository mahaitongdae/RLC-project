#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/28
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: convert_map.py
# =====================================

import matplotlib.pyplot as plt
import numpy as np

from utils.coordi_convert import vec_convert_gps_coordi_to_intersection_coordi, convert_gps_coordi_to_intersection_coordi
from utils.truncate_gps_ref import truncate_gps_ref

LANE_WIDTH_VERTICAL = 3.2
LANE_WIDTH_HORIZONTAL = 3.5
WIDTH = 43.3
HEIGHT = 53
U_OFFSET = 3.75


def convert_map():
    left_txt = np.loadtxt('../map/roadMap_ws.txt')
    stra_txt = np.loadtxt('../map/roadMap_ww.txt')
    right_txt = np.loadtxt('../map/roadMap_wn.txt')

    left_phi_in_gps, left_x_in_gps, left_y_in_gps = left_txt[:, 3], left_txt[:, 4], left_txt[:, 5]
    stra_phi_in_gps, stra_x_in_gps, stra_y_in_gps = stra_txt[:, 3], stra_txt[:, 4], stra_txt[:, 5]
    right_phi_in_gps, right_x_in_gps, right_y_in_gps = right_txt[:, 3], right_txt[:, 4], right_txt[:, 5]

    left_x, left_y, left_phi = vec_convert_gps_coordi_to_intersection_coordi(left_x_in_gps, left_y_in_gps, left_phi_in_gps)
    stra_x, stra_y, stra_phi = vec_convert_gps_coordi_to_intersection_coordi(stra_x_in_gps, stra_y_in_gps, stra_phi_in_gps)
    right_x, right_y, right_phi = vec_convert_gps_coordi_to_intersection_coordi(right_x_in_gps, right_y_in_gps, right_phi_in_gps)

    np.save('../map/left_ref.npy', np.array([left_x, left_y, left_phi]), allow_pickle=True)
    np.save('../map/straight_ref.npy', np.array([stra_x, stra_y, stra_phi]), allow_pickle=True)
    np.save('../map/right_ref.npy', np.array([right_x, right_y, right_phi]), allow_pickle=True)
    truncate_gps_ref()

def chect_converted_map():
    left_ref = np.load('../map/left_ref.npy')
    left_x, left_y, left_phi = left_ref[0], left_ref[1], left_ref[2]
    straight_ref = np.load('../map/straight_ref.npy')
    straight_x, straight_y, straight_phi = straight_ref[0], straight_ref[1], straight_ref[2]
    right_ref = np.load('../map/right_ref.npy')
    right_x, right_y, right_phi = right_ref[0], right_ref[1], right_ref[2]
    plt.plot(left_x, left_y, 'r')
    plt.plot(straight_x, straight_y, 'g')
    plt.plot(right_x, right_y, 'b')

    print(left_phi, straight_phi, right_phi)
    plt.show()

def plot_raw_map():
    DL1_txt = np.loadtxt('../map/DL1.txt')
    DL2_txt = np.loadtxt('../map/DL2.txt')
    DU_txt = np.loadtxt('../map/DU.txt')
    DR1_txt = np.loadtxt('../map/DR1.txt')
    DR2_txt = np.loadtxt('../map/DR2.txt')
    DL1_phi_in_gps, DL1_x_in_gps, DL1_y_in_gps = DL1_txt[:, 4], DL1_txt[:, 5], DL1_txt[:, 6]
    DL2_phi_in_gps, DL2_x_in_gps, DL2_y_in_gps = DL2_txt[:, 4], DL2_txt[:, 5], DL2_txt[:, 6]
    DU_phi_in_gps, DU_x_in_gps, DU_y_in_gps = DU_txt[:, 4], DU_txt[:, 5], DU_txt[:, 6]
    DR1_phi_in_gps, DR1_x_in_gps, DR1_y_in_gps = DR1_txt[:, 4], DR1_txt[:, 5], DR1_txt[:, 6]
    DR2_phi_in_gps, DR2_x_in_gps, DR2_y_in_gps = DR2_txt[:, 4], DR2_txt[:, 5], DR2_txt[:, 6]

    DL1_x, DL1_y, DL1_phi = vec_convert_gps_coordi_to_intersection_coordi(DL1_x_in_gps, DL1_y_in_gps, DL1_phi_in_gps)
    DL2_x, DL2_y, DL2_phi = vec_convert_gps_coordi_to_intersection_coordi(DL2_x_in_gps, DL2_y_in_gps, DL2_phi_in_gps)
    DU_x, DU_y, DU_phi = vec_convert_gps_coordi_to_intersection_coordi(DU_x_in_gps, DU_y_in_gps, DU_phi_in_gps)
    DR1_x, DR1_y, DR1_phi = vec_convert_gps_coordi_to_intersection_coordi(DR1_x_in_gps, DR1_y_in_gps, DR1_phi_in_gps)
    DR2_x, DR2_y, DR2_phi = vec_convert_gps_coordi_to_intersection_coordi(DR2_x_in_gps, DR2_y_in_gps, DR2_phi_in_gps)

    stop_R_phi_in_gps, stop_R_x_in_gps, stop_R_y_in_gps = 265.57, 21269691.5790374, 3448681.9698518
    stop_D_phi_in_gps, stop_D_x_in_gps, stop_D_y_in_gps = 355.89, 21269673.6260485, 3448653.15383318
    stop_L_phi_in_gps, stop_L_x_in_gps, stop_L_y_in_gps = 85.91, 21269648.2236826, 3448676.67873713
    stop_U_phi_in_gps, stop_U_x_in_gps, stop_U_y_in_gps = 174.98, 21269666.9961371, 3448709.80470448

    stop_R_x, stop_R_y, stop_R_phi = convert_gps_coordi_to_intersection_coordi(stop_R_x_in_gps, stop_R_y_in_gps, stop_R_phi_in_gps)
    stop_D_x, stop_D_y, stop_D_phi = convert_gps_coordi_to_intersection_coordi(stop_D_x_in_gps, stop_D_y_in_gps, stop_D_phi_in_gps)
    stop_L_x, stop_L_y, stop_L_phi = convert_gps_coordi_to_intersection_coordi(stop_L_x_in_gps, stop_L_y_in_gps, stop_L_phi_in_gps)
    stop_U_x, stop_U_y, stop_U_phi = convert_gps_coordi_to_intersection_coordi(stop_U_x_in_gps, stop_U_y_in_gps, stop_U_phi_in_gps)



    plt.plot(DL1_x, DL1_y, 'r')
    plt.plot(DL2_x, DL2_y, 'g')
    plt.plot(DU_x, DU_y, 'b')
    plt.plot(DR1_x, DR1_y, 'b')
    plt.plot(DR2_x, DR2_y, 'b')
    plt.plot([-100, 100], [0., 0.], linewidth=1)
    plt.plot([-100, 100], [-LANE_WIDTH_VERTICAL, -LANE_WIDTH_VERTICAL], linewidth=1, linestyle='--')
    plt.plot([-100, 100], [LANE_WIDTH_VERTICAL, LANE_WIDTH_VERTICAL], linewidth=1, linestyle='--')
    plt.plot([-100, 100], [-1/2 * LANE_WIDTH_VERTICAL, -1/2 * LANE_WIDTH_VERTICAL], linewidth=1, linestyle='--')
    plt.plot([-100, 100], [1/2 * LANE_WIDTH_VERTICAL, 1/2 * LANE_WIDTH_VERTICAL], linewidth=1, linestyle='--')
    plt.plot([-100, 100], [-2 * LANE_WIDTH_VERTICAL, -2  * LANE_WIDTH_VERTICAL], linewidth=1, linestyle='--')
    plt.plot([-100, 100], [2  * LANE_WIDTH_VERTICAL, 2 * LANE_WIDTH_VERTICAL], linewidth=1, linestyle='--')
    plt.plot([-100, 100], [-3 * LANE_WIDTH_VERTICAL, -3 * LANE_WIDTH_VERTICAL], linewidth=1, linestyle='--')
    plt.plot([-100, 100], [3 * LANE_WIDTH_VERTICAL, 3 * LANE_WIDTH_VERTICAL], linewidth=1, linestyle='--')
    plt.plot([-100, 100], [-4 * LANE_WIDTH_VERTICAL, -4 * LANE_WIDTH_VERTICAL], linewidth=1, linestyle='--')
    plt.plot([-100, 100], [4 * LANE_WIDTH_VERTICAL, 4 * LANE_WIDTH_VERTICAL], linewidth=1, linestyle='--')
    plt.plot([0., 0.], [-100., 100.], linewidth=1)
    plt.plot([LANE_WIDTH_HORIZONTAL, LANE_WIDTH_HORIZONTAL], [-100., 100.], linewidth=1, linestyle='--')#[LANE_WIDTH_HORIZONTAL, LANE_WIDTH_HORIZONTAL]

    plt.plot([-1 * LANE_WIDTH_HORIZONTAL, -1 * LANE_WIDTH_HORIZONTAL], [-100., 100.], linewidth=1, linestyle='--')
    plt.plot([1/2 * LANE_WIDTH_HORIZONTAL, 1/2 * LANE_WIDTH_HORIZONTAL], [-100., 100.], linewidth=1,
             linestyle='--')  # [LANE_WIDTH_HORIZONTAL, LANE_WIDTH_HORIZONTAL]

    plt.plot([-1/2 * LANE_WIDTH_HORIZONTAL, -1/2 * LANE_WIDTH_HORIZONTAL], [-100., 100.], linewidth=1, linestyle='--')

    plt.xlim([-50, 50])
    plt.ylim([-50, 50])

    ax = plt.axes()
    ax.add_patch(plt.Rectangle((-WIDTH / 2, -HEIGHT / 2),
                               WIDTH, HEIGHT+U_OFFSET,
                               edgecolor='black',
                               facecolor='none'))

    def plot_phi_line(x, y, phi, color='b'):
        line_length = 10
        x_forw, y_forw = x + line_length * np.cos(phi * np.pi / 180.), \
                         y + line_length * np.sin(phi * np.pi / 180.)
        plt.scatter(x, y,  s=40, marker='D')
        plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=3)
    plot_phi_line(stop_R_x, stop_R_y, stop_R_phi)
    plot_phi_line(stop_D_x, stop_D_y, stop_D_phi)
    plot_phi_line(stop_L_x, stop_L_y, stop_L_phi)
    plot_phi_line(stop_U_x, stop_U_y, stop_U_phi)

    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    plot_raw_map()



