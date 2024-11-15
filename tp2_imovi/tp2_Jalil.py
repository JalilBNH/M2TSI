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