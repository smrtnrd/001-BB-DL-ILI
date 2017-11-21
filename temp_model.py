#coding=utf-8

from __future__ import print_function

try:
    import os
except:
    pass

try:
    import sys
except:
    pass

try:
    import logging
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import pandas as pd
except:
    pass

try:
    from pandas import DataFrame
except:
    pass

try:
    from pandas import concat
except:
    pass

try:
    from random import randrange
except:
    pass

try:
    from pandas import Series
except:
    pass

try:
    from pandas import datetime
except:
    pass

try:
    from numpy import concatenate
except:
    pass

try:
    from matplotlib import pyplot
except:
    pass

try:
    from sklearn.metrics import mean_squared_error
except:
    pass

try:
    from helpers import load_pandas
except:
    pass

try:
    from helpers import series_to_supervised
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from keras.layers import Dense
except:
    pass

try:
    from keras.layers import LSTM
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform, conditional
except:
    pass

try:
    from keras.callbacks import EarlyStopping, ModelCheckpoint
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional

"""
Data providing function:

"""
logger = logging.getLogger("Load_data")
logger.info('START: LOADING THE DATA FOR TRAINING')
logger.info(
    '===============================================================================')
logger.info(
    '===============================================================================')

processed_dir = os.path.join(os.getcwd(), 'data')
input_filepath = processed_dir + \
    "/interim/001-BB-CDC_ILI_2010-2015_US_STATES-DATA_interim.pickle"

logger.info(
    '1. Loading ILI information from the following filepath: %s', input_filepath)
scaled = load_pandas(input_filepath)

# specify the number of lag weeks
n_states = 48
n_years = 3
n_weeks = 4
n_features = 6  # features
logger.info("n_weeks {}, n_features {}".format(n_weeks, n_features))
logger.info("Reframed Dataset")
reframed = series_to_supervised(scaled, n_weeks, 1)
logger.info(
    '===============================================================================')

# split into train and test sets
values = reframed.values
n_train_weeks = n_years * 48 * n_years * n_states
logger.info("n_train_weeks : {}".format(n_train_weeks))
train = values[:n_train_weeks, :]
test = values[n_train_weeks:, :]

#click.echo(reframed.head())
# split into input and outputs
n_obs = n_weeks * n_features * n_states
logger.info("Number of observation {}".format(n_obs))

x_train, y_train = train[:, :n_obs], train[:, -n_features]
x_test, y_test = test[:, :n_obs], test[:, -n_features]
logger.info("x_train shape:  ({}, {})".format(
    x_train.shape[0], x_train.shape[1]))
logger.info("y_train shape:  ({})".format(y_train.shape[0]))

logger.info(
    '===============================================================================')

# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], n_weeks, n_features))
x_test = x_test.reshape((x_test.shape[0],  n_weeks, n_features))

logger.info("x_train shape:  ({}, {}, {})".format(
    x_train.shape[0], x_train.shape[1], x_train.shape[2]))
logger.info("test_X shape:  ({}, {}, {})".format(
    x_test.shape[0], x_test.shape[1], x_test.shape[2]))
logger.info(
    '===============================================================================')



def keras_fmin_fnct(space):

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
    logger.info(
        '===============================================================================')
    logger.info(
        '===============================================================================')

    processed_dir = os.path.join(os.getcwd(), 'models')
    output_filepath = processed_dir + "/keras_LTSM200_D1.hdf5"

    model = Sequential()
    model.add(LSTM(200, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1))
    model.add(Activation('relu'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    checkpointer = ModelCheckpoint(filepath=output_filepath,
                                   verbose=2,
                                   save_best_only=True)

    model.compile(loss='mae', metrics=['accuracy'],
                  optimizer=space['optimizer'])

    history = model.fit(x_train, y_train,
                        batch_size=space['batch_size'],
                        epochs=50,
                        verbose=2,
                        #validation_split=0.8,
                        validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test, verbose=2)
    logger.info('Test accuracy: %s', acc)

    logger.info(
        '===============================================================================')
    logger.info('END: CREATING MODEL')

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    processed_dir = os.path.join(os.getcwd(), 'reports', 'figures')

    fig1 = pyplot.figure()
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('model accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'test'], loc='upper left')
    #pyplot.show()
    output_filepath = processed_dir + "/model_accuracy.png"
    fig1.savefig(output_filepath)

    fig = pyplot.figure()
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    #pyplot.show()
    output_filepath = processed_dir + "/model_train_vs_validation_loss.png"
    fig.savefig(output_filepath)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'optimizer': hp.choice('optimizer', ['rmsprop', 'adam', 'sgd']),
        'batch_size': hp.choice('batch_size', [100, 300, 500]),
    }
