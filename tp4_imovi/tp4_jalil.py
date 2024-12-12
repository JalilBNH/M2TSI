import matplotlib.pyplot as plt
import numpy as np
import pykitti
np.set_printoptions(suppress=True, precision=6)
np.set_printoptions(edgeitems=30, linewidth=120)
basedir, date, drive, frame, min_dist = './datasets/KITTI_SAMPLE_RAW/KITTI_SAMPLE/RAW', '2011_09_26', '0009', 0, 5
dataset = pykitti.raw(basedir, date, drive, frames=range(0, 5, 1))
points_cloud = dataset.get_velo(0)
points_cloud[:, -1] = 1
T = dataset.calib.T_cam2_velo
K_cam2 = dataset.calib.K_cam2
K = np.identity(4)
K[0:-1, 0:-1] = K_cam2
P = K @ T
ind = np.argwhere(points_cloud[:, 0] >= 5)
new_points = points_cloud[ind]
pts_2D = P @ new_points.T.squeeze()
pts_2D[0] /= pts_2D[2]
pts_2D[1] /= pts_2D[2]
ind1 = np.argwhere((pts_2D[0, :] > 0) & (pts_2D[0, :] < 1242) & (pts_2D[1, :] > 0) & (pts_2D[1, :] < 375))
processed_pts_2D = pts_2D[:, ind1]
ind_3D = ind.squeeze()[ind1]
plt.figure(figsize=(15,10))
plt.imshow(dataset.get_cam2(0))
plt.scatter(processed_pts_2D[0], processed_pts_2D[1], s=1.0, alpha=0.8, c=points_cloud[ind_3D, 0].squeeze(), cmap='jet'), plt.axis('off'), plt.show()