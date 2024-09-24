import os 
import numpy as np
import cv2

def load_datasets(datasets_dir, multi_class=True, read_from_stubs=False, save_stub=False):
    """Given a directory containing all our datasets, return our train and validation data in 2 numpy array.

    Args:
        datasets_dir (str): path/to/dataset_dir
        multi_class (bool, optional): Choose if you do multi-class or binary classification. Defaults to True.
        read_from_stubs (bool, optional): Read data from npy file (faster). Defaults to False.
        save_stub (bool, optional): Save a npy file. Defaults to False.

    Returns:
        np.array: train and validation datasets in np.array
    """
    if read_from_stubs and (os.path.exists(os.path.join(datasets_dir, 'dataset_01.npy')) and os.path.exists(os.path.join(datasets_dir, 'dataset_02.npy'))):
        with open(os.path.join(datasets_dir, 'dataset_01.npy'), 'rb') as f:
            dataset_01 = np.load(f)
        with open(os.path.join(datasets_dir, 'dataset_02.npy'), 'rb') as f:
            dataset_02 = np.load(f)
        
        return dataset_01, dataset_02
            
            
    train_total_size = 0
    val_total_size = 0
    
    if multi_class:
        for folder in os.listdir(datasets_dir):
            if folder.endswith('01'):
                train_total_size  += len(os.listdir(os.path.join(datasets_dir, folder)))
            elif folder.endswith('02'):
                val_total_size += len(os.listdir(os.path.join(datasets_dir, folder)))  
    
    else:
        for folder in os.listdir(datasets_dir)[:4]:
            if folder.endswith('01'):
                train_total_size  += len(os.listdir(os.path.join(datasets_dir, folder)))
            elif folder.endswith('02'):
                val_total_size += len(os.listdir(os.path.join(datasets_dir, folder)))
        
    dataset_01 = np.empty(shape=(train_total_size, 540, 920, 3), dtype=np.float32)
    dataset_02 = np.empty(shape=(val_total_size, 540, 920, 3), dtype=np.float32)
        
    i = 0
    for folder in os.listdir(datasets_dir):
        if folder.endswith('01'):
            for file in os.listdir(os.path.join(datasets_dir, folder)):
                dataset_01[i] = cv2.imread(os.path.join(datasets_dir, folder, file)) / 255.0
                i += 1
    i = 0
    for folder in os.listdir(datasets_dir):
        if folder.endswith('02'):
            for file in os.listdir(os.path.join(datasets_dir, folder)):
                dataset_02[i] = cv2.imread(os.path.join(datasets_dir, folder, file)) / 255.0
                i += 1
    
    if save_stub:
        with open(os.path.join(datasets_dir, 'dataset_01.npy'), 'wb') as f:
            np.save(f, dataset_01)
        with open(os.path.join(datasets_dir, 'dataset_02.npy'), 'wb') as f:
            np.save(f, dataset_02)

    return dataset_01, dataset_02