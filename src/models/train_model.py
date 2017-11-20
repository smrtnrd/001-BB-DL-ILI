# -*- coding: utf-8 -*-
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import logging

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat

from numpy import concatenate
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from helpers import load_pandas
from helpers import series_to_supervised

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import LSTM

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.callbacks import EarlyStopping, ModelCheckpoint

def data():
    """
    Data providing function:

    """
    logger = logging.getLogger("Load_data")
    logger.info('START: LOADING THE DATA FOR TRAINING')
    logger.info('===============================================================================')
    logger.info('===============================================================================')
    
    processed_dir = os.path.join(os.getcwd(), 'data')
    input_filepath = processed_dir + "/interim/001-BB-CDC_ILI_2010-2015_US_STATES-DATA_interim.pickle"

    logger.info('1. Loading ILI information from the following filepath: %s', input_filepath)
    scaled = load_pandas(input_filepath)
    print(scaled.head())
    # specify the number of lag weeks
    n_states = 48
    n_years = 4
    n_weeks = 12
    n_features = 6 #features
    logger.info("Lag weeks {}, Number of features {}".format(n_weeks, n_features))
    logger.info("Reframed Dataset")
    reframed = series_to_supervised(scaled, n_weeks, 1)   
    print(reframed.head())
    logger.info('===============================================================================')
    logger.info('===============================================================================')

    # split into train and test sets
    values = reframed.values
    n_train_weeks = 4 * 12 * n_years * n_states
    logger.info("Number of training weeks: {}".format(n_train_weeks))
    train = values[:n_train_weeks, :]
    test = values[n_train_weeks:, :]

    #click.echo(reframed.head())
    # split into input and outputs
    n_obs = n_weeks * n_features * n_states
    logger.info("Number of observation {}".format(n_obs))

    x_train, y_train = train[:, :n_obs], train[:, -n_features]
    x_test, y_test = test[:, :n_obs], test[:, -n_features]

    logger.info('===============================================================================')
    logger.info('===============================================================================')
    logger.info('===============================================================================')
    logger.info('===============================================================================')

    # reshape input to be 3D [samples, timesteps, features]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    logger.info("x_train shape:  ({}, {}, {})".format(x_train.shape[0],x_train.shape[1],x_train.shape[2] ))
    logger.info("test_X shape:  ({}, {}, {})".format(x_test.shape[0],x_test.shape[1],x_test.shape[2]))
    logger.info('===============================================================================')
    logger.info('===============================================================================')


    return x_train, y_train, x_test, y_test



def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    logger = logging.getLogger("create_model")
    logger.info('START: CREATING MODEL')
    logger.info('===============================================================================')
    logger.info('===============================================================================')
    model = Sequential()
    model.add(LSTM(50, input_shape = (x_train.shape[1], x_train.shape[2])))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
                                   verbose=1,
                                   save_best_only=True)


    model.compile(loss='mae', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    history = model.fit(x_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              epochs={{choice([5,10,50])}},
              verbose=2,
              validation_split=0.08,
              validation_data=(x_test, y_test))
    
    score, acc = model.evaluate(x_test, y_test, verbose=0)

    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    print('Test accuracy:', acc)

    logger.info('===============================================================================')
    logger.info('END: CREATING MODEL')

    print('Test score:', score)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def main():
    """ Prepare dataset for Deep Learning 
    """
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    

if __name__ == '__main__':
     # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='train_model.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    main()
