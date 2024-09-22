# Jalil Tp2
# Git repository to see more details and download the different features and target stubs: https://github.com/JalilBNH/M2TSI/tree/main/tp2


import numpy as np
from matplotlib import pyplot as plt
import os
from keras.api.models import Model
from keras.api.layers import Input, Dense, Activation, Flatten, LeakyReLU, BatchNormalization
from keras.api.optimizers import Adam
from keras.api.losses import binary_crossentropy, categorical_crossentropy
from keras.api.utils import to_categorical
from keras.api.models import load_model
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd

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


def extract_simple_features(image):
    """Given an image, return the mean of each channel.

    Args:
        image (np.array): image in np.array format

    Returns:
        np.array: the mean of each channel
    """
    return np.array([np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])])


def extract_features(image):
    """Given an image, return a feature vector.

    Args:
        image (np.array): image in np.array format

    Returns:
        np.array: feature vector
    """
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    epsilon = 1e-10 
    
    hist, bin = np.histogram(image_gray, bins=10)
    bin = bin[1:]
    
    fft_image = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image_gray))) + epsilon)
    hist1, bin1 = np.histogram(fft_image, bins=10)
    bin1 = bin1[1:]

    return np.concatenate([
        np.array([np.mean(image), np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])]),
        (hist*bin/np.sum(hist)),
        (hist1*bin1)/np.sum(hist1)
    ])

def create_features_vector(dataset_01=None, dataset_02=None, complex_features=True, read_from_stubs=False, save_stubs=False, stub_path=None):
    """Given our datasets return the features vector.

    Args:
        dataset_01 (np.array, optional): dataset in np.array format. Optional if you read from stubs. Defaults to None.
        dataset_02 (np.array, optional): dataset in np.array format. Optional if you read from stubs. Defaults to None.
        complex_features (bool, optional): Choose between the extract_simple_features or extract_features. Defaults to True.
        read_from_stubs (bool, optional): Read data from npy file (faster). Defaults to False.
        save_stubs (bool, optional): Save data in a npy file. Defaults to False.
        stub_path (_type_, optional): path/to/npy_file. Defaults to None.

    Returns:
        np.array: features matrix
    """
    if read_from_stubs and os.path.exists(os.path.join(stub_path, 'features_01.npy')) and os.path.exists(os.path.join(stub_path, 'features_02.npy')):
        with open(os.path.join(stub_path, 'features_01.npy'), 'rb') as f:
            features_01 = np.load(f)
        with open(os.path.join(stub_path, 'features_02.npy'), 'rb') as f:
            features_02 = np.load(f)
        return features_01, features_02
    
    
    if complex_features:
        features_01 = np.empty(shape=(dataset_01.shape[0], 24))
        features_02 = np.empty(shape=(dataset_02.shape[0], 24))
        for i, (image_1, image_2) in enumerate(zip(dataset_01, dataset_02)):
            features_01[i] = extract_features(image_1)
            features_02[i] = extract_features(image_2)
    else:
        features_01 = np.empty(shape=(dataset_01.shape[0], 3))
        features_02 = np.empty(shape=(dataset_02.shape[0], 3))
        for i, (image_1, image_2) in enumerate(zip(dataset_01, dataset_02)):
            features_01[i] = extract_simple_features(image_1)
            features_02[i] = extract_simple_features(image_2)
            
    if save_stubs:
        with open(os.path.join(stub_path, 'features_01.npy'), 'wb') as f:
            np.save(f, features_01)
        with open(os.path.join(stub_path, 'features_02.npy'), 'wb') as f:
            np.save(f, features_02)
        
    return features_01, features_02

def create_target_vector(features_01=None, features_02=None, mult_class=True, save_stub=False, read_from_stubs=False, stub_path=None):
    """Given the features matrix return the targets vectors.

    Args:
        features_01 (np.array, optional): Features matrix. Defaults to None.
        features_02 (np.array, optional): Features matrix. Defaults to None.
        mult_class (bool, optional): Choose to do binary or mult-class classification. Defaults to True.
        save_stub (bool, optional): Save data in a npy file. Defaults to False.
        read_from_stubs (bool, optional): Read data from npy file (faster). Defaults to False.
        stub_path (_type_, optional): path/to/npy_file. Defaults to None.

    Returns:
        np.array: target vectors
    """
    if read_from_stubs and os.path.exists(os.path.join(stub_path, 'target_01.npy')) and os.path.exists(os.path.join(stub_path, 'target_02.npy')):
        with open(os.path.join(stub_path, 'target_01.npy'), 'rb') as f:
            target_01 = np.load(f)
        with open(os.path.join(stub_path, 'target_02.npy'), 'rb') as f:
            target_02 = np.load(f)
        return target_01, target_02
    
    target_01 = np.zeros(features_01.shape[0])
    target_02 = np.zeros(features_02.shape[0])
    
    if mult_class:
        ind = [500, 1000, 1500]
        val = [1, 2, 3]
        for i, v in enumerate(val):
            target_01[ind[i]:] = v
            target_02[ind[i]:] = v
        target_01 = to_categorical(target_01)
        target_02 = to_categorical(target_02)

    else:
        target_01[500:] = 1
        target_02[500:] = 1
    
    if save_stub:
        with open(os.path.join(stub_path, 'target_01.npy'), 'wb') as f:
            np.save(f, target_01)
        with open(os.path.join(stub_path, 'target_02.npy'), 'wb') as f:
            np.save(f, target_02)
            
    return target_01, target_02

def create_model(complex_features=True, mult_class=True):
    """Create the DNN model.

    Args:
        complex_features (bool, optional): Choose between complex features or simple features. Defaults to True.
        mult_class (bool, optional): Choose to do binary or mult-class classification. Defaults to True.

    Returns:
        keras.model: DNN Model
    """
    
    if complex_features:
        input = Input(shape=(24,))
    else:
        input = Input(shape=(3,))
    x = Dense(units=32)(input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dense(units=16)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if mult_class:
        output = Dense(units=4, activation='softmax')(x)
    else:
        output = Dense(units=1, activation='sigmoid')(x)
    
    return Model(input, output)

    
        


def main():
    
    
    #dataset_01, dataset_02 = load_datasets()
    #features_01, features_02 = create_features_vector()
    
    features_01, features_02 = create_features_vector(
        read_from_stubs=True,
        stub_path=r'C:\Users\Jalil\Desktop\Ecole\M2TSI\tp2\datasets\Movies'
    )
    
    target_01, target_02 = create_target_vector(
        read_from_stubs=True,
        stub_path=r'C:\Users\Jalil\Desktop\Ecole\M2TSI\tp2\datasets\Movies'
    )
    
    x_train, x_test, y_train, y_test = train_test_split(features_01, target_01, train_size=0.8, shuffle=True, random_state=13)

    model = create_model()
    model.summary()
    optimizer = Adam(learning_rate=0.0005)
    
    if model.input.shape[1] > 3:
        model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=['accuracy'])
    
    history = model.fit(
        x_train,
        y_train,
        batch_size=16,
        epochs=50,
        validation_data=(x_test, y_test)
    )

    evaluation = model.evaluate(x_test, y_test)
    print(f'evaluation : {evaluation}')
    model_path = r'C:\Users\Jalil\Desktop\Ecole\M2TSI\tp2\models'
    model.save(os.path.join(model_path, 'model.h5'))
    
    excel_path = 'Jalil_Excel.xlsx'
    df = pd.DataFrame(history.history)
    df.to_excel(excel_path, index=False)    

    






if __name__ == '__main__':
    main()