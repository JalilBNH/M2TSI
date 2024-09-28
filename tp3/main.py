# Jalil Tp2
# Git repository to get more details with the dev jupyter notebook and download the features and target stubs to run the code faster: https://github.com/JalilBNH/M2TSI/tree/main/tp3
# Excel sheet legend :  - bin = binary classification
#                       - mutl = multi-class classification


import numpy as np
import os
import cv2
import sklearn
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from keras.api.models import Model
from keras.api.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Flatten, Softmax, Activation, MaxPool2D, Input, Dropout, GlobalAveragePooling2D
from keras.api.optimizers import Adam
from keras.api.losses import binary_crossentropy, categorical_crossentropy
from keras.api.utils import to_categorical
from keras.api.saving import load_model
import random
import matplotlib.style
import seaborn as sns
import pandas as pd


def load_train_val(datasets_dir, multi_class=False, new_shape=(270, 460), read_from_stubs=False, stub_path=None, save_stub=False, shuffle=False):
    
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


def load_test(datasets_dir, multi_class=False, new_shape=(270, 460), read_from_stubs=False, stub_path=None, save_stub=False, shuffle=False):
    
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

def choose_model(multi_class=False):
    if not multi_class:
        input = Input((270, 460, 3))
        x = Conv2D(filters=8, kernel_size=3, strides=1)(input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPool2D()(x)
        x = Conv2D(filters=16, kernel_size=3, strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPool2D()(x)
        x = Flatten()(x)
        x = Dense(16)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(rate = 0.5)(x)
        x = Dense(1)(x)
        output = Activation('sigmoid')(x)
        return Model(input, output)
    else:
        input = Input((270, 460, 3))
        x = Conv2D(filters=4, kernel_size=3, strides=1)(input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPool2D()(x)
        x = Conv2D(filters=8, kernel_size=3, strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPool2D()(x)
        x = Conv2D(filters=16, kernel_size=3, strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPool2D()(x)
        x = Flatten()(x)
        x = Dense(16)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(4)(x)
        output = Activation('softmax')(x)

        return Model(input, output)

def plot_confusion_matrix(y_test, y_pred, mult_class=False):
    if not mult_class:
        cm = confusion_matrix(y_test, (y_pred >= 0.5).astype(int), labels=[0, 1])
        title = 'Confusion matrix binary class'
    else:
        cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=[0, 1, 2, 3])
        title = 'Confusion matrix mult class'
    sns.heatmap(cm, annot=True, fmt='g', cmap='Greens'), plt.title(title)
    plt.xlabel('Predicted classes'), plt.ylabel('True classes')
    plt.show()
    

def main():
    MULTI_CLASS = True
    SHUFFLE = True
    BATCH_SIZE = 4
    EPOCHS = 5
    LEARNING_RATE = 0.0005
    data_path = './datasets/Movies'
    loss_fct = categorical_crossentropy if MULTI_CLASS else binary_crossentropy
    
    (x_train, y_train), (x_val, y_val) = load_train_val(
        datasets_dir=data_path,
        multi_class=MULTI_CLASS,
        shuffle=SHUFFLE
    )

    (x_test, y_test) = load_test(
        datasets_dir=data_path,
        multi_class=MULTI_CLASS,
        shuffle=SHUFFLE
    )
    
    model = choose_model(multi_class=MULTI_CLASS)
    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss=loss_fct, metrics=['accuracy'])
    
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=EPOCHS,
        epochs=BATCH_SIZE,
        validation_data=(x_val, y_val)
    )
    
    evaluation = model.evaluate(x=x_test, y=y_test)
    y_pred = model.predict(x_test)
    plot_confusion_matrix(y_test, y_pred, mult_class=MULTI_CLASS)

    
if __name__ == '__main__':
    main()