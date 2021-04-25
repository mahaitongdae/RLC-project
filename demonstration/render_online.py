import math
import time
import numpy as np
from math import cos,sin,pi
from utils.misc import TimerStat
from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL.Image import open
import xml.dom.minidom
EGO_LENGTH = 4.8
EGO_WIDTH = 2.0
STATE_OTHER_LENGTH = 4.2
STATE_OTHER_WIDTH = 1.8
SCALE = 60
SIZE = 1000

def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

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

class Render():
    def __init__(self, shared_list, lock, args):
        self.args = args
        self.shared_list = shared_list
        self.lock = lock
        self.task = self.args.task
        self.model_only_test = self.args.model_only_test
        self.step_old = -1
        self.acc_timer = TimerStat()
        self._load_xml()
        self.red_img = self._read_png('utils/Rendering/red.png')
        self.green_img = self._read_png('utils/Rendering/green.png')
        self.GL_TEXTURE_RED = glGenTextures(1)
        self.GL_TEXTURE_GREEN = glGenTextures(1)
        left_construct_traj = np.load('./map/left_ref.npy')
        straight_construct_traj = np.load('./map/straight_ref.npy')
        right_construct_traj = np.load('./map/right_ref.npy')
        self.ref_path_all = {'left': left_construct_traj, 'straight': straight_construct_traj,
                             'right': right_construct_traj}


    def run(self):
        self._opengl_start()

    def _read_png(self, path):
        im = open(path)
        ix, iy, image = im.size[0], im.size[1], im.tobytes("raw", "RGBA", 0, -1)
        return ix, iy, image

    def _opengl_start(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitContextProfile(GLUT_CORE_PROFILE)
        glutInitWindowSize(SIZE,SIZE)
        glutInitWindowPosition(460, 0)
        glutCreateWindow('Crossroad')
        glutDisplayFunc(self.render)
        glutIdleFunc(self.render)
        # glutTimerFunc(20,self.render, 0)
        glutMainLoop()

    def _load_xml(self, path="./utils/sumo_files/a.net.xml"):
        dom_obj = xml.dom.minidom.parse(path)
        element_obj = dom_obj.documentElement
        self.sub_element_edge = element_obj.getElementsByTagName("edge")
        self.sub_element_junction = element_obj.getElementsByTagName("junction")
        self.sub_element_tlLogic = element_obj.getElementsByTagName("tlLogic")

    def _draw_map(self, scale):
        sub_element_edge = self.sub_element_edge
        sub_element_junction = self.sub_element_junction

        def bit_pattern(*args):
            args = list(args[:16])
            args = args + [0] * (16 - len(args))
            base = 0
            for arg in args:
                base = base << 1
                base += bool(arg)
            return base

        for i in range(len(sub_element_edge)):
            if sub_element_edge[i].getAttribute('function') == '':
                sub_element_lane = sub_element_edge[i].getElementsByTagName("lane")
                if sub_element_edge[i].getAttribute('id') in ['1i','1o','3i','3o']:
                    vertical = True
                else:
                    vertical = False
                for j in range(len(sub_element_lane)):
                    shape = sub_element_lane[j].getAttribute("shape")
                    type = str(sub_element_lane[j].getAttribute("allow"))
                    type_2 = str(sub_element_lane[j].getAttribute("disallow"))
                    try:
                        width = float(sub_element_lane[j].getAttribute("width"))
                    except:
                        width = 3.5
                    shape_list = shape.split(" ")
                    for k in range(len(shape_list) - 1):
                        shape_point_1 = shape_list[k].split(",")
                        shape_point_2 = shape_list[k + 1].split(",")
                        shape_point_1[0] = float(shape_point_1[0])
                        shape_point_1[1] = float(shape_point_1[1])
                        shape_point_2[0] = float(shape_point_2[0])
                        shape_point_2[1] = float(shape_point_2[1])
                        # 道路顶点生成
                        dx1 = shape_point_2[0] - shape_point_1[0]
                        dy1 = shape_point_2[1] - shape_point_1[1]
                        v1 = np.array([-dy1, dx1])
                        absdx = abs(shape_point_1[0] - shape_point_2[0])
                        absdy = abs(shape_point_1[1] - shape_point_2[1])
                        if math.sqrt(absdx * absdx + absdy * absdy) > 0:
                            v3 = v1 / math.sqrt(absdx * absdx + absdy * absdy) * width * 0.5
                            [x1, y1] = ([shape_point_1[0], shape_point_1[1]] + v3) / scale
                            [x2, y2] = ([shape_point_1[0], shape_point_1[1]] - v3) / scale
                            [x4, y4] = ([shape_point_2[0], shape_point_2[1]] + v3) / scale
                            [x3, y3] = ([shape_point_2[0], shape_point_2[1]] - v3) / scale  # 0.0176 * v3
                            glBegin(GL_POLYGON)  # 开始绘制单车道
                            if type == 'pedestrian':
                                glColor3f(0.663, 0.663, 0.663)
                            elif type == 'bicycle':
                                glColor3f(0.545, 0.279, 0.074)
                            elif type_2 == 'all':
                                glColor3f(0.1333, 0.545, 0.1333)
                            else:
                                glColor3f(0.0, 0.0, 0.0)

                            glVertex2f(x1, y1)
                            glVertex2f(x2, y2)
                            glVertex2f(x3, y3)
                            glVertex2f(x4, y4)
                            glEnd()

                            if type == '' and type_2 != 'all':
                                glLineWidth(1.0)
                                glLineStipple(3, bit_pattern(
                                    0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 1, 1,
                                    1, 1, 1, 1,
                                ))
                                glEnable(GL_LINE_STIPPLE)
                                glBegin(GL_LINES)
                                glColor3f(1.0, 1.0, 1.0)
                                glVertex2f(x2, y2)
                                glVertex2f(x3, y3)
                                if not j == sub_element_lane.length - 1:
                                    glVertex2f(x1, y1)
                                    glVertex2f(x4, y4)
                                glEnd()
                                glDisable(GL_LINE_STIPPLE)
                                glLineWidth(10.0)
                                glBegin(GL_LINES)
                                glColor3f(1.0, 1.0, 1.0)
                                glVertex2f(x3, y3)
                                glVertex2f(x4, y4)
                                glEnd()
                                glLineWidth(2.0)
                                glBegin(GL_LINES)
                                glColor3f(0.9, 0.7, 0.05)
                                if j == sub_element_lane.length - 1:
                                    if not vertical:
                                        glVertex2f(x1, y1 - 0.005)
                                        glVertex2f(x4, y4 - 0.005)
                                        glVertex2f(x1, y1 + 0.005)
                                        glVertex2f(x4, y4 + 0.005)
                                    if vertical:
                                        glVertex2f(x1 - 0.005, y1)
                                        glVertex2f(x4 - 0.005, y4)
                                        glVertex2f(x1 + 0.005, y1)
                                        glVertex2f(x4 + 0.005, y4)
                                glEnd()
                                glBegin(GL_LINES)
                                glColor3f(0.8275, 0.8275, 0.8275)
                                if not vertical:
                                    if j == 3:
                                        glVertex2f(x2, y2)
                                        glVertex2f(x3, y3)
                                if vertical:
                                    if j == 2:
                                        glVertex2f(x2, y2)
                                        glVertex2f(x3, y3)
                                glEnd()




        for i in range(len(sub_element_junction)):
            shape = sub_element_junction[i].getAttribute("shape")
            shape_list = shape.split(" ")

            glLineWidth(1)
            glBegin(GL_POLYGON)
            glColor3f(0.0, 0.0, 0.0)
            for k in range(len(shape_list)):
                shape_point = shape_list[k].split(",")
                if shape_point[0] != '':
                    glVertex2f((float(shape_point[0]) / scale) * 1, (float(shape_point[1]) / scale) * 1)
            glEnd()
            # glutTimerFunc(20,self.render,0)


    def _draw_zebra(self, loc, width, length, scale, shape, single_height=0.8):
        glLineWidth(1)
        glColor3f(0.8275, 0.8275, 0.8275)
        if shape == 'vertical':
            for i in range(int(length/single_height)):
                glBegin(GL_POLYGON)
                x1, y1 = (loc - width / 2) / scale, (2 * i * single_height) / scale
                x2, y2 = (loc + width / 2) / scale, (2 * i * single_height) / scale
                x3, y3 = (loc + width / 2) / scale, ((2 * i + 1) * single_height) / scale
                x4, y4 = (loc - width / 2) / scale, ((2 * i + 1) * single_height) / scale
                glVertex2f(x1, y1)
                glVertex2f(x2, y2)
                glVertex2f(x3, y3)
                glVertex2f(x4, y4)
                glEnd()
                glBegin(GL_POLYGON)
                x1, y1 = (loc - width / 2) / scale, -((2 * i + 2) * single_height) / scale
                x2, y2 = (loc + width / 2) / scale, -((2 * i + 2) * single_height) / scale
                x3, y3 = (loc + width / 2) / scale, -((2 * i + 1) * single_height) / scale
                x4, y4 = (loc - width / 2) / scale, -((2 * i + 1) * single_height) / scale
                glVertex2f(x1, y1)
                glVertex2f(x2, y2)
                glVertex2f(x3, y3)
                glVertex2f(x4, y4)
                glEnd()
        elif shape == 'horizontal':
            for i in range(int(length/single_height)):
                glBegin(GL_POLYGON)
                y1, x1 = (loc - width / 2) / scale, (2 * i * single_height) / scale
                y2, x2 = (loc + width / 2) / scale, (2 * i * single_height) / scale
                y3, x3 = (loc + width / 2) / scale, ((2 * i + 1) * single_height) / scale
                y4, x4 = (loc - width / 2) / scale, ((2 * i + 1) * single_height) / scale
                glVertex2f(x1, y1)
                glVertex2f(x2, y2)
                glVertex2f(x3, y3)
                glVertex2f(x4, y4)
                glEnd()
                glBegin(GL_POLYGON)
                y1, x1 = (loc - width / 2) / scale, -((2 * i + 2) * single_height) / scale
                y2, x2 = (loc + width / 2) / scale, -((2 * i + 2) * single_height) / scale
                y3, x3 = (loc + width / 2) / scale, -((2 * i + 1) * single_height) / scale
                y4, x4 = (loc - width / 2) / scale, -((2 * i + 1) * single_height) / scale
                glVertex2f(x1, y1)
                glVertex2f(x2, y2)
                glVertex2f(x3, y3)
                glVertex2f(x4, y4)
                glEnd()

    def _plot_reference(self, task, highlight_index, scale):
        glPointSize(2.0)
        glBegin(GL_POINTS)
        for i in range(self.ref_path_all[task].shape[0]):
            if i != highlight_index:
                glColor3f(0.5, 0.5, 0.5)
                for j in range(self.ref_path_all[task].shape[2]):
                    x = self.ref_path_all[task][i][0][j] / scale
                    y =  self.ref_path_all[task][i][1][j] / scale
                    glVertex2f(x, y)
        for i in range(self.ref_path_all[task].shape[0]):
            if i == highlight_index:
                glColor3f(0.486, 0.99, 0.0)
                for j in range(self.ref_path_all[task].shape[2]):
                    x = self.ref_path_all[task][i][0][j] / scale
                    y =  self.ref_path_all[task][i][1][j] / scale
                    glVertex2f(x, y)
        glEnd()

    def _text(self, str, column, loc):
        if loc == 'left':
            glRasterPos3f(-1, 1.00 - 0.05 * column, 0.0)
        elif loc == 'right':
            glRasterPos3f(0.4, 1.00 - 0.05 * column, 0.0)
        n = len(str)
        for i in range(n):
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(str[i]))

    def render(self, real_x=0, real_y=0, scale=SCALE, **kwargs):
        LOC_X = -real_x / scale
        LOC_Y = -real_y / scale
        glClearColor(0.753, 0.753, 0.753, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(LOC_X, LOC_Y, 0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # draw map
        self._draw_map(scale)
        self._draw_zebra(-17, 6, 10, scale, 'vertical')
        self._draw_zebra(17, 6, 10, scale, 'vertical')
        self._draw_zebra(26, 6, 6, scale, 'horizontal')
        self._draw_zebra(-22, 6, 6, scale, 'horizontal')

        v_light = self.shared_list[13]
        if v_light == 0:
            self._texture_light(self.green_img, (-15, 30), 'U', scale)
            self._texture_light(self.green_img, (7, -29), 'D', scale)
            self._texture_light(self.red_img, (-24.5, -21), 'L', scale)
            self._texture_light(self.red_img, (21.5, 13), 'R', scale)
        elif v_light != 0:
            self._texture_light(self.red_img, (-15, 30), 'U', scale)
            self._texture_light(self.red_img, (7, -29), 'D', scale)
            self._texture_light(self.green_img, (-24.5, -21), 'L', scale)
            self._texture_light(self.green_img, (21.5, 13), 'R', scale)

        # draw ref
        path_index = self.shared_list[12]
        self._plot_reference(self.task, path_index, scale)

        # draw vehicles
        def draw_vehicle(x, y, a, l, w, scale, color='o', details=True):
            RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
            RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
            LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
            glBegin(GL_POLYGON)
            if color == 'o':
                glColor3f(1.0, 0.647, 0.0)
            elif color == 'y':
                glColor3f(1.0, 1.0, 0.878)
            elif color == 'blue':
                glColor3f(0.2, 0.3, 0.9)
            elif color == 'g':
                glColor3f(0.5, 1.0, 0.0)
            glVertex2f((RU_x + x) / scale, (RU_y + y) / scale)
            glVertex2f((RD_x + x) / scale, (RD_y + y) / scale)
            glVertex2f((LD_x + x) / scale, (LD_y + y) / scale)
            glVertex2f((LU_x + x) / scale, (LU_y + y) / scale)
            glEnd()

            if details:

                RU1_x, RU1_y, _ = rotate_coordination(l / 4, w * 4 / 10, 0, -a)
                RD1_x, RD1_y, _ = rotate_coordination(l / 4, - w * 4 / 10, 0, -a)
                LU1_x, LU1_y, _ = rotate_coordination(l / 12, w * 3 / 10, 0, -a)
                LD1_x, LD1_y, _ = rotate_coordination(l / 12, -w * 3 / 10, 0, -a)

                glBegin(GL_POLYGON)
                glColor3f(0.0,0.0,0.0)
                glVertex2f((RU1_x + x) / scale, (RU1_y + y) / scale)
                glVertex2f((RD1_x + x) / scale, (RD1_y + y) / scale)
                glVertex2f((LD1_x + x) / scale, (LD1_y + y) / scale)
                glVertex2f((LU1_x + x) / scale, (LU1_y + y) / scale)
                glEnd()

                RU2_x, RU2_y, _ = rotate_coordination(-l / 3, w * 4 / 10, 0, -a)
                RD2_x, RD2_y, _ = rotate_coordination(-l / 3, - w * 4 / 10, 0, -a)
                LU2_x, LU2_y, _ = rotate_coordination(-l / 6, w * 3 / 10, 0, -a)
                LD2_x, LD2_y, _ = rotate_coordination(-l / 6, -w * 3 / 10, 0, -a)

                glBegin(GL_POLYGON)
                glColor3f(0.0, 0.0, 0.0)
                glVertex2f((RU2_x + x) / scale, (RU2_y + y) / scale)
                glVertex2f((RD2_x + x) / scale, (RD2_y + y) / scale)
                glVertex2f((LD2_x + x) / scale, (LD2_y + y) / scale)
                glVertex2f((LU2_x + x) / scale, (LU2_y + y) / scale)
                glEnd()

            return LU_x, LU_y, LD_x, LD_y


        def plot_phi_line(x, y, phi, color, scale):
            line_length = 5
            x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                             y + line_length * sin(phi * pi / 180.)
            glLineWidth(1.0)
            glBegin(GL_LINES)
            if color == 'o':
                glColor3f(1.0, 0.647, 0.0)
            elif color == 'y':
                glColor3f(1.0, 1.0, 0.878)
            elif color == 'blue':
                glColor3f(0.2, 0.3, 0.9)
            glVertex2f(x / scale,y / scale)
            glVertex2f(x_forw / scale,y_forw / scale)
            glEnd()


        # ego vehicle
        state_ego = self.shared_list[9].copy()
        ego_x = state_ego['GaussX']
        ego_y = state_ego['GaussY']
        ego_phi = state_ego['Heading']
        decision = self.shared_list[8].copy()
        acc = decision['a_x']

        # draw safety shield
        ss_flag = self.shared_list[14]
        if ss_flag:
            _, _, _, _ = draw_vehicle(ego_x, ego_y, ego_phi, EGO_LENGTH + 1, EGO_WIDTH+1,
                                      scale, color='green', details=False)

        LU_x, LU_y, LD_x, LD_y = draw_vehicle(ego_x, ego_y, ego_phi, EGO_LENGTH, EGO_WIDTH, scale, color='o')
        plot_phi_line(ego_x,ego_y,ego_phi,'o',scale)
        if acc < 0:
            glPointSize(3.0)
            glBegin(GL_POINTS)
            glColor3f(1.0, 0.0, 0.0)
            glVertex2f((LU_x + ego_x) / scale, (LU_y + ego_y) / scale)
            glVertex2f((LD_x + ego_x) / scale, (LD_y + ego_y) / scale)
            glEnd()

        # if not self.model_only_test:
        #
        #     model_action_x = state_ego['model_x_in_model_action']
        #     model_action_y = state_ego['model_y_in_model_action']
        #     model_action_phi = state_ego['model_phi_in_model_action']
        #     plot_phi_line(model_action_x, model_action_y, model_action_phi, 'blue', scale)
        #     _,_,_,_ = draw_vehicle(model_action_x, model_action_y, model_action_phi, EGO_LENGTH, EGO_WIDTH, scale, 'blue')

        state_other = self.shared_list[4].copy()

        # plot cars
        for veh in state_other:
            veh_x = veh['x']
            veh_y = veh['y']
            veh_phi = veh['phi']
            veh_l = STATE_OTHER_LENGTH
            veh_w = STATE_OTHER_WIDTH
            plot_phi_line(veh_x, veh_y, veh_phi, 'y', scale)
            _,_,_,_ = draw_vehicle(veh_x, veh_y, veh_phi, veh_l, veh_w, scale, color='y')

        traj_value = self.shared_list[11]
        for i in range(len(traj_value)):
            if not path_index == i:
                glColor3f(0.3, 0.3, 0.3)
            else:
                glColor3f(0.3, 0.5, 0.1)
            str1 = 'Path ' + str(i) + ' reward: ' + str(traj_value[i][0])[:7]
            self._text(str1, i + 1, 'left')
            str2 = 'Path ' + str(i) + ' collision risk: ' + str(traj_value[i][1])[:7]
            self._text(str2, i + 1, 'right')

        glutSwapBuffers()

        glDisable(GL_BLEND)
        glDisable(GL_LINE_SMOOTH)
        glDisable(GL_POLYGON_SMOOTH)


    def _texture_light(self, img, loc, edge, scale, size=(8, 3)):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.GL_TEXTURE_RED)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, 3, img[0], img[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, img[2])

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if edge == 'U' or edge=='D':
            x1, y1 = loc[0] / scale, loc[1] / scale
            x2, y2 = (loc[0] + size[0]) / scale, loc[1] / scale
            x3, y3 = loc[0] / scale, (loc[1] + size[1]) / scale
            x4, y4 = (loc[0] + size[0]) / scale, (loc[1] + size[1]) / scale
        elif edge == 'L' or edge=='R':
            x1, y1 = loc[0] / scale, loc[1] / scale
            x2, y2 = (loc[0] + size[1]) / scale, loc[1] / scale
            x3, y3 = loc[0] / scale, (loc[1] + size[0]) / scale
            x4, y4 = (loc[0] + size[1]) / scale, (loc[1] + size[0]) / scale
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        if edge == 'U':
            glTexCoord2f(0.0, 0.0)
            glVertex3f(x4, y4, 0.0)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(x3, y3, 0.0)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(x1, y1, 0.0)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(x2, y2, 0.0)
        elif edge == 'D':
            glTexCoord2f(0.0, 0.0)
            glVertex3f(x1, y1, 0.0)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(x2, y2, 0.0)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(x4, y4, 0.0)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(x3, y3, 0.0)
        elif edge == 'L':
            glTexCoord2f(0.0, 0.0)
            glVertex3f(x3, y3, 0.0)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(x1, y1, 0.0)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(x2, y2, 0.0)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(x4, y4, 0.0)
        elif edge == 'R':
            glTexCoord2f(0.0, 0.0)
            glVertex3f(x2, y2, 0.0)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(x4, y4, 0.0)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(x3, y3, 0.0)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(x1, y1, 0.0)

        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_ALPHA_TEST)
        glDisable(GL_BLEND)




if __name__ == '__main__':
    path_index = 0
    shared_list = [[1],[1],[1],[1],[{'x':0.0, 'y':10.0,'phi':135.0}],[1],[1],[1],[1],{'GaussX':0.0, 'GaussY':50.0,'Heading':135.0},[],[[0,1,2],[0,1,2],[0,1,2],[0,1,2]],1,0]
    render = Render(shared_list, None, 'right')
    render.run()
