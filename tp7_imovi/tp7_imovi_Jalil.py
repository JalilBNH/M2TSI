import cv2
from matplotlib import pyplot as plt
import numpy as np
import random
import pykitti

np.set_printoptions(suppress=True, precision=6)
np.set_printoptions(edgeitems=30, linewidth=120)
basedir, date, drive = './datasets/KITTI_SAMPLE_RAW/KITTI_SAMPLE/RAW', '2011_09_26', '0009'
dataset = pykitti.raw(basedir, date, drive, frames=range(0, 445, 1))
img01, img02 = np.array(dataset.get_cam2(0)), np.array(dataset.get_cam3(2))
K = dataset.calib.K_cam2
s = cv2.SIFT_create()
kp_01, desc_01 = s.detectAndCompute(img01, None)
kp_02, desc_02 = s.detectAndCompute(img02, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(desc_01,desc_02)
l1 = np.array([kp_01[match.queryIdx].pt for match in matches])
l2 = np.array([kp_02[match.trainIdx].pt for match in matches])
E, mask = cv2.findEssentialMat(l1, l2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
inlier_l1 = np.array([l1[i] for i in range(len(l1)) if mask[i]])
inlier_l2 = np.array([l2[i] for i in range(len(l2)) if mask[i]])
retval, R, t, mask_pose, pts_3d_h = cv2.recoverPose(E, inlier_l1, inlier_l2, K, distanceThresh=100)
pts_3d = pts_3d_h / pts_3d_h[3, :]
Z = pts_3d[2, :]
mask_pts = (pts_3d[2, :] >= np.percentile(Z, 5)) & (pts_3d[2, :]  <= np.percentile(Z, 95))
new_pts_3d = pts_3d[:, mask_pts]
new_l1 = inlier_l1[mask_pts]
new_l2 = inlier_l2[mask_pts]

Z = new_pts_3d[2,:]

plt.figure(figsize=(12,14))
plt.imshow(img01), plt.axis('off')
plt.scatter(new_l1[:, 0], new_l1[:, 1], c=Z, s=7, cmap='jet_r') 
plt.tight_layout()
plt.show()
plt.figure(figsize=(12,14))
plt.imshow(img02), plt.axis('off')
plt.scatter(new_l2[:, 0], new_l2[:, 1], c=Z, s=7, cmap='jet_r') 
plt.tight_layout()
plt.show()

