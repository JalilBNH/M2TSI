import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def make_projective(theta, t, alpha_u, alpha_v, u0, v0):
    K = np.array([
        [alpha_u, 0, u0],
        [0, alpha_v, v0],
        [0, 0, 1]
    ])
    
    Rx, Ry, Rz = theta
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(Rx), -np.sin(Rx)],
        [0, np.sin(Rx), np.cos(Rx)]
    ])
    R_y = np.array([
        [np.cos(Ry), 0, np.sin(Ry)],
        [0, 1, 0],
        [-np.sin(Ry), 0, np.cos(Ry)]
    ])
    R_z = np.array([
        [np.cos(Rz), -np.sin(Rz), 0],
        [np.sin(Rz), np.cos(Rz), 0],
        [0, 0, 1]
    ])
    
    R = R_z @ R_y @ R_x
    T = np.array([[t[0]], [t[1]], [t[2]]])
    RT = np.hstack((R, T))
    P = K @ RT
    return P

def projection(P, X):
    x_projected = P @ X.T
    return x_projected[:2, :] / x_projected[2, :]

def dlt(points_3D, points_2D):
 
    N = points_3D.shape[1]
    A = []
    
    for i in range(N):
        X, Y, Z, _ = points_3D[:, i]
        u, v, _ = points_2D[:, i]

        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])

    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)  

    return P

def evaluate(P, P_dlt):
    error = np.abs((P / P[-1, -1]) - (P_dlt / P_dlt[-1, -1]))

    print(f'Mean error : {np.mean(error)}')
    print(f'Std error : {np.std(error)}')

nb_pts = 8

Rx = 0.8 * np.pi/2
Ry = -1.8 * np.pi/2
Rz = np.pi/5
Tx = 100 
Ty = 0
Tz = 1500
a_u = 557.0943
a_v = 712.9824
u0 = 326.3819
v0 = 298.6679

np.random.seed(10) 
pts_3D = np.random.randint(low=-480, high=480, size=(nb_pts, 3))
pts_3D = np.hstack((pts_3D, np.ones((nb_pts, 1))))  
print('---------------------------------')
print("Points 3D :\n", pts_3D)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pts_3D[:, 0], pts_3D[:, 1], pts_3D[:, 2], marker='o')

for i, (x, y, z) in enumerate(pts_3D[:, :-1]):
    ax.text(x, y, z, f"P{i+1}")

plt.title("points 3D")
plt.show()

P = make_projective((Rx, Ry, Rz), (Tx, Ty, Tz), a_u, a_v, u0, v0)
pts_2D = projection(P, pts_3D)
pts_2D_homo = np.vstack((pts_2D, np.ones((1, pts_2D.shape[1]))))

print('---------------------------------')
print(f'P : \n{P}')

fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(pts_2D[0, :], pts_2D[1, :], marker='o')

for i, (x, y) in enumerate(zip(pts_2D[0], pts_2D[1])):
    ax.text(x, y, f"P{i+1}")
ax.grid()
plt.title('Points 2D')
plt.show()

P_dlt = dlt(pts_3D.T, pts_2D_homo)
print('---------------------------------')
print("P_DLT :\n", P_dlt)

print('---------------------------------')
evaluate(P, P_dlt)

noise = np.random.normal(0, np.sqrt(1), (2, nb_pts))
pts_2D_noise = pts_2D + noise
pts_2D_noise = np.vstack((pts_2D_noise, np.ones((1, pts_2D_noise.shape[1]))))

P_dlt_noise = dlt(pts_3D.T, pts_2D_noise)
print('---------------------------------')
evaluate(P, P_dlt_noise)