import os 
import cv2
import numpy as np
import random
import sklearn
from keras.api.utils import to_categorical
from matplotlib import pyplot as plt

def load_train_val(datasets_dir, multi_class=True, new_shape=(270, 460), read_from_stubs=False, stub_path=None, save_stub=False, shuffle=False):
    
    if read_from_stubs and stub_path is not None and len(os.listdir(stub_path)) > 0:
        with open(os.path.join(stub_path, 'ds01_train.npy'), 'rb') as f:
            ds01_train = np.load(f)
        with open(os.path.join(stub_path, 'ds01_val.npy'), 'rb') as f:
            ds01_val = np.load(f)
        with open(os.path.join(stub_path, 'y01_train.npy'), 'rb') as f:
            y01_train = np.load(f)
        with open(os.path.join(stub_path, 'y01_val.npy'), 'rb') as f:
            y01_val = np.load(f)
        
        return ds01_train, y01_train, ds01_val, y01_val 
    
    folders = os.listdir(datasets_dir) if multi_class else os.listdir(datasets_dir)[:4]

    train_total_size = 0
    for folder in folders:
        if folder.endswith('01'):
            train_total_size  += len(os.listdir(os.path.join(datasets_dir, folder)))
    
    ds01_train = np.empty(shape=(int(0.8*train_total_size), new_shape[0], new_shape[1], 3), dtype=np.float32)
    ds01_val = np.empty(shape=(int(0.2*train_total_size), new_shape[0], new_shape[1], 3), dtype=np.float32)
    y01_train = np.empty(shape=(int(0.8*train_total_size), 1))
    y01_val = np.empty(shape=(int(0.2*train_total_size), 1))
    idx_01_train = 0
    idx_01_val = 0
    y01_train_class = 0
    y01_val_class = 0
    
    for folder in folders:
        len_folder = len(os.listdir(os.path.join(datasets_dir, folder)))  
        
        images_path = os.listdir(os.path.join(datasets_dir, folder))
        random.shuffle(images_path)
        if folder.endswith('01'):
            
            for i, file in enumerate(images_path[:int(0.8*len_folder)]):
                ds01_train[i+idx_01_train] = cv2.resize(cv2.imread(os.path.join(datasets_dir, folder, file)) / 255.0, new_shape[::-1])
                y01_train[i+idx_01_train] = y01_train_class
            idx_01_train += int(0.8*len_folder)
            y01_train_class += 1
            
            for i, file in enumerate(images_path[int(0.8*len_folder):]):
                ds01_val[i+idx_01_val] = cv2.resize(cv2.imread(os.path.join(datasets_dir, folder, file)) / 255.0, new_shape[::-1])
                y01_val[i+idx_01_val] = y01_val_class
            idx_01_val += int(0.2*len_folder)
            y01_val_class += 1
    
    if multi_class:
        y01_train = to_categorical(y01_train)
        y01_val = to_categorical(y01_val)
        
    if save_stub:
        with open(os.path.join(stub_path, 'ds01_train.npy'), 'wb') as f:
            np.save(f, ds01_train)
        with open(os.path.join(stub_path, 'ds01_val.npy'), 'wb') as f:
            np.save(f, ds01_val)
        with open(os.path.join(stub_path, 'y01_train.npy'), 'wb') as f:
            np.save(f, y01_train)
        with open(os.path.join(stub_path, 'y01_val.npy'), 'wb') as f:
            np.save(f, y01_val)
    
        
    if not shuffle:
        return (ds01_train, y01_train), (ds01_val, y01_val)
    else:
        return sklearn.utils.shuffle(ds01_train, y01_train), sklearn.utils.shuffle(ds01_val, y01_val)
    


def load_test(datasets_dir, multi_class=True, new_shape=(270, 460), read_from_stubs=False, stub_path=None, save_stub=False, shuffle=False):
    
    if read_from_stubs and stub_path is not None and len(os.listdir(stub_path)) > 0:
        with open(os.path.join(stub_path, 'ds02.npy'), 'rb') as f:
            ds02 = np.load(f)
        with open(os.path.join(stub_path, 'y02.npy'), 'rb') as f:
            y02 = np.load(f)
        return(ds02, y02)
    
    folders = os.listdir(datasets_dir) if multi_class else os.listdir(datasets_dir)[:4]
    test_total_size = 0
    for folder in folders:
        if folder.endswith('02'):
            test_total_size  += len(os.listdir(os.path.join(datasets_dir, folder)))
    
    ds02 = np.empty(shape=(test_total_size, new_shape[0], new_shape[1], 3), dtype=np.float32)
    y02 = np.empty(shape=(test_total_size, 1))
    
    idx_val = 0
    y02_val_class = 0
    for folder in folders:
        len_folder = len(os.listdir(os.path.join(datasets_dir, folder)))
        if folder.endswith('02'):
            for file in os.listdir(os.path.join(datasets_dir, folder)):
                ds02[idx_val] = cv2.resize(cv2.imread(os.path.join(datasets_dir, folder, file)) / 255.0, new_shape[::-1])
                y02[idx_val] = y02_val_class
                idx_val += 1
            y02_val_class += 1
    
    if multi_class:
        y02 = to_categorical(y02)
    
    if save_stub:
        with open(os.path.join(stub_path, 'ds02.npy'), 'wb') as f:
            np.save(f, ds02)
        with open(os.path.join(stub_path, 'y02.npy'), 'wb') as f:
            np.save(f, y02)

    if not shuffle:
        return (ds02, y02)
    else:
        return sklearn.utils.shuffle(ds02, y02)


def visualize_data_sample(x, y, random=False):
    if random:
        ind = np.random.permutation(x.shape[0])
        plt.figure(figsize=(20,6))
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(x[ind[i], :, :, ::-1]), plt.title(f'class : {str(y[ind[i]])}, ind : {ind[i]}'), plt.axis('off')

    else:
        plt.figure(figsize=(20,6))
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(x[i, :, :, ::-1]), plt.title(f'class : {str(y[i])}, ind : {i}'), plt.axis('off')
