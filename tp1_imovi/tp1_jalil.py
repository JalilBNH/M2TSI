import numpy as np
from matplotlib import pyplot as plt

# 1.4 

# Question 1

l1 = np.array([[1, 1, -5]])
l2 = np.array([[4, -5, 7]])

x = np.cross(l1, l2)
print(f'x : {x}')

print(f'verification : l1t.x = {np.dot(l1, x.T).squeeze()}, l2t.x = {np.dot(l2, x.T).squeeze()}')
print(x.shape)

x_cart = np.array([val / x[0, -1] for val in x[0, :-1]])
print(f'x_cart : {x_cart}')

    

# Question 2

l1 = np.array([[1, 2, 1]])
l2 = np.array([[3, 6, -2]])

x = np.cross(l1, l2)
print(f'x : {x}')
# Notre point se forme à l'infini car notre coordonné homogène = 0

# Question 3

pt1 = [[1, 3, 1]] # On ajoute les coordonnées homogènes
pt2 = [[2, 7, 1]]
l = np.cross(pt1, pt2)
print(f'la droite l passe par les deux point : {l}')

# Transformations géométriques

# Question 1 :  
# M représente représentation la rotation et l'homothétie tandis que Q représente la translation.
# et M[3,:] représente les coordonnées homogènes


# Question 2 : 

M = np.array([[0, -1, 0, 2], [1, 0, 0, 5], [0, 0, 1, 4], [0, 0, 0, 1]])
A = np.array([[3, 4, 5, 1]])

print(f'pre transform : {np.linalg.norm(A)}')
A_transformed = np.dot(M, A.T)
print(f'post transform : {np.linalg.norm(A_transformed)}')

def transform_m(angles, translation, scale_factor):

    t = np.identity(n=4)
    s = np.identity(n=4)
    
    
    rx = np.array([[1, 0, 0, 0],
                  [0, np.cos(np.radians(angles[0])), -np.sin(np.radians(angles[0])), 0],
                  [0, np.sin(np.radians(angles[0])), np.cos(np.radians(angles[0])), 0],
                  [0, 0, 0, 1]])
    
    ry = np.array([[np.cos(np.radians(angles[1])), 0, np.sin(np.radians(angles[1])), 0],
                  [0, 1, 0, 0],
                  [-np.sin(np.radians(angles[1])), 0, np.cos(np.radians(angles[1])), 0],
                  [0, 0, 0, 1]])
    
    rz = np.array([[np.cos(np.radians(angles[2])), -np.sin(np.radians(angles[2])), 0, 0],
                  [np.sin(np.radians(angles[2])), np.cos(np.radians(angles[2])), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    
    for i, val_params in enumerate(zip(angles, translation)):
        trans = val_params[1]
        t[i, -1] = trans
        
        
    
    r = np.dot(rx, np.dot(ry, rz))
    s[:-1, :-1] = s[:-1, :-1]*scale_factor
    
    return np.dot(t, np.dot(r, s))

a = np.array([[-0.5, -0.5, 0, 1]])
b = np.array([[-0.5, 0.5, 0, 1]])
c = np.array([[0.5, -0.5, 0, 1]])
d = np.array([[0.5, 0.5, 0, 1]])


pts = [a, b, c, d]
transformed_pts = []
m = transform_m(angles=[45, 0, 45], translation=[0.5, 0.5, 2], scale_factor=2)
for pt in pts:
    transformed_pts.append(np.dot(m, pt.T).T)
    
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for pt, transformed_pt in zip(pts, transformed_pts):
    ax.scatter3D(pt[0, 0], pt[0, 1], pt[0, 2], color='blue')
    ax.scatter3D(transformed_pt[0, 0], transformed_pt[0, 1], transformed_pt[0, 2], color='red') # En rouge, nous avons les points transformées.
plt.legend()
plt.show()
    
    


