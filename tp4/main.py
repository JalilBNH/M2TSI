import os 
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import sys
sys.path.append('../')
from utils import visualize_data_sample
from sklearn.utils import shuffle
from keras.api.models import Model, Sequential
from keras.api.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, Flatten, Softmax, Activation, MaxPool2D, Input, Dropout, ReLU, RandomBrightness, RandomContrast, RandomFlip, RandomZoom, GaussianNoise
from keras.api.optimizers import Adam
from keras.api.losses import binary_crossentropy, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.api.metrics import Accuracy
import tensorflow as tf
import keras


def preprocess_output(y_data):
    y_data = y_data[:, 0:2]
    y_data[:, 0] = np.abs(y_data[:, 0])
    y_data[:, 0] = np.clip(y_data[:, 0], 0, 5)
    y_data[:, 1] = (y_data[:, 1] >= 0).astype(int)
    return y_data



def main():
    x_data = np.load('../datasets/tp4/pipeline_images.npy')
    y_data = np.load('../datasets/tp4/pipeline_predections.npy')
    
    y_data = preprocess_output(y_data)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True)
    
    data_augmentation = Sequential([
        RandomContrast(factor=0.01),
        RandomBrightness(factor=0.01),
        RandomZoom(height_factor=(0.1, 0.2), width_factor=(0.1,0.2)),
    ])

    input = Input((228, 308, 3))
    x = data_augmentation(input)
    
    x = Conv2D(filters=16, kernel_size=3, strides=1)(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)
    
    x = Conv2D(filters=32, kernel_size=3, strides=2)(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)
    
    x = Conv2D(filters=64, kernel_size=3, strides=1)(x)
    x = ReLU()(x)
    x = MaxPool2D()(x)
    
    x = Flatten()(x)
    
    x = Dense(64)(x)
    x = ReLU()(x)
    
    classification = Dense(1, activation='sigmoid', name='classification')(x)
    regression = Dense(1, name='regression')(x)

    model = Model(inputs=input, outputs=[classification, regression])
    
    model.summary()
    opt = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=opt,
        loss={'classification': 'binary_crossentropy', 'regression': 'mean_squared_error'},
        metrics={'classification': 'accuracy', 'regression': 'mean_absolute_error'},
    )
    
    history = model.fit(
        x=x_train,
        y=[y_train[:, 1].reshape((-1,1)), y_train[:, 0].reshape((-1,1))],
        batch_size=16, 
        epochs=20
    )
    
    model.evaluate(
        x=x_test,
        y=[y_test[:, 1].reshape((-1,1)), y_test[:, 0].reshape((-1,1))]
    )

if __name__ == '__main__':
    main()