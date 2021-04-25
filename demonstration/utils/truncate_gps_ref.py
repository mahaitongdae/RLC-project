import matplotlib.pyplot as plt
import numpy as np


def truncate_gps_ref():
    left = np.load('../map/left_ref.npy')
    straight = np.load('../map/straight_ref.npy')
    right = np.load('../map/right_ref.npy')
    plt.figure()
    for i in range(len(left[0])):
        if left[1][i] >= - 54:
            start = i
            break
    for i in range(len(right[0])):
        if left[0][i] <= -51:
            stop = i
            break
    left_real = left[:, start:stop]
    plt.plot(left_real[0], left_real[1])
    np.save('../map/left_ref_cut.npy', left_real)
    for i in range(len(straight[0])):
        if straight[1][i] >= - 54:
            start = i
            break
    for i in range(len(straight[0])):
        if straight[1][i] >= 51:
            stop = i
            break
    straight_real = straight[:, start:stop]
    plt.plot(straight_real[0], straight_real[1])
    np.save('../map/straight_ref_cut.npy', straight_real)
    for i in range(len(right[0])):
        if right[1][i] >= - 54:
            start = i
            break
    for i in range(len(right[0])):
        if right[0][i] >= 51:
            stop = i
            break
    right_real = right[:, start:stop]
    plt.plot(right_real[0], right_real[1])
    plt.show()
    np.save('../map/right_ref_cut.npy', right_real)


if __name__ == '__main__':
    truncate_gps_ref()