import os 
import numpy as np
from matplotlib import pyplot as plt
import cv2


work_dir = 'c:/Users/Jalil/Desktop/Ecole/M2TSI/tp2/'
dataset_toystory_path = os.path.join(work_dir, 'datasets/toy_story_1_01')
dataset_ernest_path = os.path.join(work_dir, 'datasets/Movies/ernest_celestine_01')

image_mat = np.empty((1000, 540, 920, 3))


for i, file in enumerate(os.listdir(os.path.join(work_dir, dataset_toystory_path))):
    path = os.path.join(work_dir, dataset_toystory_path, file)
    
    image_mat[i] = cv2.imread(path) / 255.0
    
    

cv2.imshow('test', image_mat[12])
cv2.waitKey(0)
cv2.destroyAllWindows()