import numpy as np

data = np.load('hierarchical_decision/results/2021-03-08-15-16-25/data_across_all_episodes.npy',allow_pickle=True)
# np.array([v_x, v_y, r, x, y, phi, steer, a_x, delta_y,
#                                         delta_phi, delta_v, cal_time, ref_index, beta, path_values, ss_time])
a = 1