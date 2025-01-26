import cv2
from matplotlib import pyplot as plt
import numpy as np
import pykitti

np.set_printoptions(suppress=True, precision=6, edgeitems=30, linewidth=120)
basedir, date, drive = './datasets/KITTI_SAMPLE_RAW/KITTI_SAMPLE/RAW', '2011_09_26', '0009'
dataset = pykitti.raw(basedir, date, drive, frames=range(0, 50, 1))
oxts_positions = np.array([oxts_data.T_w_imu[:3, -1] for oxts_data in dataset.oxts])
x, z, y = oxts_positions[:, 0], oxts_positions[:, 1], oxts_positions[:, 2]
trajectory = [np.array([0, 0, 0])]
T_cam2_cam3 = np.linalg.inv(dataset.calib.T_cam3_velo) @ dataset.calib.T_cam2_velo
V1 = T_cam2_cam3[:3, 3]

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
for ind1 in range(0, 49): 
    lp = np.array(dataset.get_cam2(ind1))
    rp = np.array(dataset.get_cam3(ind1))
    lc = np.array(dataset.get_cam2(ind1 + 1))
    kp_lp, desc_lp = sift.detectAndCompute(lp, None)
    kp_lc, desc_lc = sift.detectAndCompute(lc, None)
    matches_lp = bf.match(desc_lc, desc_lp)
    l1 = np.array([kp_lc[m.queryIdx].pt for m in matches_lp])
    l2 = np.array([kp_lp[m.trainIdx].pt for m in matches_lp])
    E1, _ = cv2.findEssentialMat(l1, l2, method=cv2.FM_8POINT + cv2.FM_RANSAC)
    _, _, t2, _ = cv2.recoverPose(E1, l1, l2)  
    kp_rp, desc_rp = sift.detectAndCompute(rp, None)
    matches_lp_rp = bf.match(desc_lp, desc_rp)
    l1 = np.array([kp_lp[m.queryIdx].pt for m in matches_lp_rp])
    l2 = np.array([kp_rp[m.trainIdx].pt for m in matches_lp_rp])
    E2, _ = cv2.findEssentialMat(l1, l2, method=cv2.FM_8POINT + cv2.FM_RANSAC)
    _, _, t3, _ = cv2.recoverPose(E2, l1, l2)  
    A = np.column_stack((t2.flatten(), -t3.flatten()))
    S = np.linalg.lstsq(A, -V1, rcond=None)[0]
    s1, s2 = np.abs(S)
    t2_scaled = s1 * t2.flatten() 
    t3_scaled = s2 * t3.flatten()  
    pose = trajectory[-1] + t2_scaled  
    trajectory.append(pose)
trajectory = np.array(trajectory)

plt.subplot(121), plt.plot(trajectory[:, 0], trajectory[:, 2]), plt.title("Trajectoire calculée")
plt.subplot(122), plt.plot(x, z), plt.title('Trajectoire dataset')
plt.show()
