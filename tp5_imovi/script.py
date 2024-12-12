import matplotlib.pyplot as plt
import numpy as np
import pykitti

np.set_printoptions(suppress=True, precision=6)
np.set_printoptions(edgeitems=30, linewidth=120)
basedir, date, drive, frame, min_dist = './datasets/KITTI_SAMPLE_RAW/KITTI_SAMPLE/RAW', '2011_09_26', '0009', 0, 5
dataset = pykitti.raw(basedir, date, drive, frames=range(0, 5, 1))
points_cloud = dataset.get_velo(0)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(points_cloud[:, 0], points_cloud[:, 1], points_cloud[:, 2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()