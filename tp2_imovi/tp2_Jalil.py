import numpy as np
from matplotlib import pyplot as plt

def make_projective(theta, t, ax, ay, x0, y0):
    K = np.array([[ax, 0, x0],
                  [0, ay, y0],
                  [0, 0, 1]])
    
    theta = np.radians(theta) 
     
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    
    T = np.array([[0], [0], [t]])
    
    RT = np.hstack((R_x, T))
    
    P = K @ RT
    return P


def projection(P, X):
    X_h = np.vstack((X, np.ones((1, X.shape[1]))))
    x_h = P @ X_h
    
    return x_h[:2, :] / x_h[2, :]

cube_points = np.array([[0, 1, 1, 0, 0, 1, 1, 0],
                        [0, 0, 1, 1, 0, 0, 1, 1],
                        [0, 0, 0, 0, 1, 1, 1, 1]])

P = make_projective(20, 2, 800, 800, 250, 250)
projected_points = projection(P, cube_points)

plt.figure()
plt.scatter(projected_points[0, :], projected_points[1, :], c='blue', marker='o')
plt.show()