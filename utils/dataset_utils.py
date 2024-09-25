import os 
import numpy as np
import cv2

def load_datasets(datasets_dir, multi_class=True, read_from_stubs=False, stub_path=None, save_stub=False):    
    """Given a directory containing all our datasets, return our train and validation data in 2 numpy array.

    Args:
        datasets_dir (str): path/to/dataset_dir
        multi_class (bool, optional): Choose if you do multi-class or binary classification. Defaults to True.
        read_from_stubs (bool, optional): Read data from npy file (faster). Defaults to False.
        stub_path (str, optional):path/to/stubs. Defaults to None.
        save_stub (bool, optional): Save a npy file. Defaults to False.


    Returns:
        np.array: train test and validation datasets in np.array
    """
    if read_from_stubs and stub_path is not None and len(os.listdir(stub_path)) == 3:
        with open(os.path.join(stub_path, 'ds_01_train.npy'), 'rb') as f:
            dataset_01_train = np.load(f)
        with open(os.path.join(stub_path, 'ds_01_val.npy'), 'rb') as f:
            dataset_01_val = np.load(f)
        with open(os.path.join(stub_path, 'ds_02.npy'), 'rb') as f:
            dataset_02 = np.load(f)
        
        return dataset_01_train, dataset_01_val, dataset_02
    
    train_total_size = 0
    test_total_size = 0
    if multi_class:
        for folder in os.listdir(datasets_dir):
            if folder.endswith('01'):
                train_total_size  += len(os.listdir(os.path.join(datasets_dir, folder)))
            elif folder.endswith('02'):
                test_total_size += len(os.listdir(os.path.join(datasets_dir, folder)))      
    else:
        for folder in os.listdir(datasets_dir)[:4]:
            if folder.endswith('01'):
                train_total_size  += len(os.listdir(os.path.join(datasets_dir, folder)))
            elif folder.endswith('02'):
                test_total_size += len(os.listdir(os.path.join(datasets_dir, folder)))
    dataset_01_train = np.empty(shape=(int(0.8*train_total_size), 540, 920, 3), dtype=np.float32)
    dataset_01_val = np.empty(shape=(int(0.2*train_total_size), 540, 920, 3), dtype=np.float32)
    dataset_02 = np.empty(shape=(test_total_size, 540, 920, 3), dtype=np.float32)
    
    idx_test = 0
    idx_train = 0
    idx_val = 0
    
    folders = os.listdir(datasets_dir) if multi_class else os.listdir(datasets_dir)[:4]

    for folder in folders:
        len_folder = len(os.listdir(os.path.join(datasets_dir, folder)))  
        if folder.endswith('01'):
            for i, file in enumerate(os.listdir(os.path.join(datasets_dir, folder))[:int(0.8*len_folder)]):
                dataset_01_train[i+idx_train] = cv2.imread(os.path.join(datasets_dir, folder, file)) / 255.0
            idx_train += int(0.8*len_folder)
            for i, file in enumerate(os.listdir(os.path.join(datasets_dir, folder))[int(0.8*len_folder):]):
                dataset_01_val[i+idx_test] = cv2.imread(os.path.join(datasets_dir, folder, file)) / 255.0
            idx_test += int(0.2*len_folder)
        if folder.endswith('02'):
            for file in os.listdir(os.path.join(datasets_dir, folder)):
                dataset_02[idx_val] = cv2.imread(os.path.join(datasets_dir, folder, file)) / 255.0
                idx_val += 1
                           
    if save_stub:
        with open(os.path.join(stub_path, 'ds_01_train.npy'), 'wb') as f:
            np.save(f, dataset_01_train)
        with open(os.path.join(stub_path, 'ds_01_val.npy'), 'wb') as f:
            np.save(f, dataset_01_val)
        with open(os.path.join(stub_path, 'ds_02.npy'), 'wb') as f:
            np.save(f, dataset_02)
            
    return dataset_01_train, dataset_01_val, dataset_02
    