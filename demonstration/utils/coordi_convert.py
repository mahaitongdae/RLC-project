#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/28
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: coordi_convert.py
# =====================================

import math

import numpy as np

ROTATE_ANGLE = 3.2
X_IN_GPS_OFFSET = -13.5
Y_IN_GPS_OFFSET = -1.25
stop_R_phi_in_gps, stop_R_x_in_gps, stop_R_y_in_gps = 265.57, 21269691.5790374, 3448681.9698518
stop_D_phi_in_gps, stop_D_x_in_gps, stop_D_y_in_gps = 355.89, 21269673.6260485, 3448653.15383318
stop_L_phi_in_gps, stop_L_x_in_gps, stop_L_y_in_gps = 85.91, 21269648.2236826, 3448676.67873713
stop_U_phi_in_gps, stop_U_x_in_gps, stop_U_y_in_gps = 174.98, 21269666.9961371, 3448709.80470448
ORIGIN_X_IN_GPS, ORIGIN_Y_IN_GPS = (stop_L_x_in_gps+stop_R_x_in_gps)/2.+0.5, (stop_D_y_in_gps+stop_U_y_in_gps)/2.-2.2


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    coordi_rotate_d_in_rad = coordi_rotate_d * math.pi / 180
    transformed_x = orig_x * math.cos(coordi_rotate_d_in_rad) + orig_y * math.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * math.sin(coordi_rotate_d_in_rad) + orig_y * math.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    if transformed_d > 180:
        while transformed_d > 180:
            transformed_d = transformed_d - 360
    elif transformed_d <= -180:
        while transformed_d <= -180:
            transformed_d = transformed_d + 360
    else:
        transformed_d = transformed_d
    return transformed_x, transformed_y, transformed_d


def shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def convert_gps_coordi_to_intersection_coordi(x, y, phi):
    phi_in_anticlockwise = -(phi-90.)
    if phi_in_anticlockwise>180:
        phi_in_anticlockwise -=360
    elif phi_in_anticlockwise <=-180:
        phi_in_anticlockwise += 360
    trans_x, trans_y, trans_phi = shift_and_rotate_coordination(x, y, phi_in_anticlockwise,
                                  ORIGIN_X_IN_GPS, ORIGIN_Y_IN_GPS, ROTATE_ANGLE)
    return trans_x, trans_y, trans_phi


def vec_shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def vec_rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    coordi_rotate_d_in_rad = coordi_rotate_d * np.pi / 180
    transformed_x = orig_x * np.cos(coordi_rotate_d_in_rad) + orig_y * np.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * np.sin(coordi_rotate_d_in_rad) + orig_y * np.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    transformed_d = np.where(transformed_d > 180., transformed_d - 360., transformed_d)
    transformed_d = np.where(transformed_d <= -180., transformed_d + 360., transformed_d)
    return transformed_x, transformed_y, transformed_d


def vec_shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = vec_shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = vec_rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def vec_convert_gps_coordi_to_intersection_coordi(x, y, phi):
    phi_in_anticlockwise = -(phi-90.)
    phi_in_anticlockwise = np.where(phi_in_anticlockwise>180, phi_in_anticlockwise-360, phi_in_anticlockwise)
    phi_in_anticlockwise = np.where(phi_in_anticlockwise<-180, phi_in_anticlockwise+360, phi_in_anticlockwise)
    trans_x, trans_y, trans_phi = vec_shift_and_rotate_coordination(x, y, phi_in_anticlockwise,
                                  ORIGIN_X_IN_GPS, ORIGIN_Y_IN_GPS, ROTATE_ANGLE)
    return trans_x, trans_y, trans_phi


def convert_rear_to_center(x_rear_axle, y_rear_axle, heading, bias=1.4):
    """ RTK传来的自车位置不是在车辆正中心（车辆后轴）,需要换算到车辆的正中心"""
    x_modified = x_rear_axle + bias * math.cos(heading * math.pi / 180)
    y_modified = y_rear_axle + bias * math.sin(heading * math.pi / 180)
    return x_modified, y_modified, heading


def convert_center_to_rear(x_center, y_center, heading, bias=1.4):
    "used before sending info to radar system"
    x_rear = x_center - bias * math.cos(heading * math.pi / 180)
    y_rear = y_center - bias * math.sin(heading * math.pi / 180)
    return x_rear, y_rear, heading

