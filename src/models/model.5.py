import warnings

import pydot
import graphviz

# Take a look at the raw data :
import pandas as pd
from pandas import Series
from pandas import DataFrame
from pandas import read_csv

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import matplotlib
# be able to save images on server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from math import sqrt
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
import sys
import errno
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

import keras
from keras.layers import Input, Convolution1D, Dense, MaxPooling1D, Flatten
from keras.layers import LSTM
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
# be able to save images on server
# matplotlib.use('Agg')
import time
import datetime

from keras.models import load_model

import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" %
                          self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


class RData:
    def __init__(self, path, n_weeks=26):
        self.path = path
        self.data = {}
        # load dataset
        self.data['raw'] = self.load_data()

        # config
        self.n_weeks = n_weeks
        self.n_features = int(len(self.data['raw'][0].columns))
        print("number of features: {}".format(self.n_features))

        # scale data
        self.scaler = preprocessing.MinMaxScaler()
        self.scale()
        # reframe data
        self.reframe()

        # self.state_list_name = self.data.state.unique()
        self.split_data()
        # print(self.n_features)

    # Return specific data
    def __getitem__(self, index):
        return self.data[index]

    # load dataset
    def load_data(self):
        raw = read_csv(self.path)
        raw = raw.fillna(0)
        # print(raw['0'].head())
        # raw = raw.drop(["0"],  axis = 1)
        # print(raw.head())

        # transform column names
        raw.columns = map(str.lower, raw.columns)
        # raw.rename(columns={'weekend': 'date'}, inplace=True)
        latitudeList = raw.latitude.unique()
        longitudeList = raw.longitude.unique()
        data_list = list()
        cell_label = list()
        for la in latitudeList:
            for lo in longitudeList:
                data = raw[(raw.latitude == la) & (raw.longitude == lo)]
                if(len(data) == 260):
                    select = [
                        #'date',
                        #'year',
                        #'month',
                        #'week',
                        #'week_temp',
                        #'week_prcp',
                        #'latitude',
                        #'longitude',
                        'mean_ili',
                        #'ili_activity_label',
                        #'ili_activity_group'
                    ]
                    # One Hot Encoding
                    data = pd.get_dummies(data[select])
                    # print(data.head(1))
                    data_list.append(data)
                    cell_label.append('lat {} - long {}'.format(la, lo))
                    print("The data for latitude {} and longitude {} contains {} rows".format(
                        la, lo, len(data)))
        self.data['cell_labels'] = cell_label
        print("The are {} cell in the data".format(len(data_list)))
        return data_list

    # convert series to supervised learning
    @staticmethod
    def series_to_supervised(df, n_in=26, n_out=26, dropnan=True):
        from pandas import concat
        data = DataFrame(df)
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        input_list, target_list = list(), list()
        input_names, target_names = list(), list()

        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            input_list.append(df.shift(i))
            input_names += [('var%d(t-%d)' % (j + 1, i))
                             for j in range(n_vars)]

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            target_list.append(df.shift(-i))
            if i == 0:
                target_names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                target_names += [('var%d(t+%d)' % (j + 1, i))
                                  for j in range(n_vars)]

        # put it all together
        samples = concat(input_list, axis=1)
        samples.columns = input_names

        targets = concat(target_list, axis=1)
        targets.columns = target_names

        # drop rows with NaN values
        if dropnan:
            targets.fillna(-1, inplace=True)
            samples.fillna(-1, inplace=True)

        supervised = [samples, targets]

        return supervised
    
    # convert series to supervised learning
    @staticmethod
    def series_to_reframed(data, n_in=26, n_out=26, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        print("length :{}".format(len(agg)))
        agg.columns = names
        
        # drop rows with NaN values
        if dropnan:
            agg.fillna(-1, inplace=True)
        return agg
    # frame a sequence as a supervised learning problem
    @staticmethod
    def _series_to_supervised(data, lag=26):
        from pandas import concat
        df = DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag + 1)]
        columns.append(df)
        df = concat(columns, axis=1)
        df = df.fillna(0)
        return df

    # normalize
    def scale(self):
        scaled = list()
        for df in self.data['raw']:
            scaled_df = self.scaler.fit_transform(df)
            scaled_df = pd.DataFrame(scaled_df, columns=df.columns.values)
            scaled.append(scaled_df)
        self.data['scaled'] = scaled

    # create a differenced series
    @staticmethod
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)

    def reframe(self):
        # specify the number of lag_weeks
        reframed,supervised  = list(), list()
        for scaled in self.data['scaled']:
            supervised.append(self.series_to_supervised(scaled))
            # frame as supervised learning
            reframed.append(self.series_to_reframed(scaled))
        
        self.data['supervised'] = supervised
        self.data['reframed'] = reframed

    # Return specific data
    def split_data(self):
        n_train_weeks = 7 * 26

        # split into train and test sets
        train_X, train_y = list(), list()
        test_X, test_y = list(), list()

        train_n_X, train_n_y = list(), list()
        test_n_X, test_n_y = list(), list()
        for reframed in self.data['reframed']:
            # sample, target = reframed[0], reframed[1]
            values = reframed.values
            train = values[:n_train_weeks, :]
            self.data['train'] = train
            
            # train_target = target.values[:n_train_weeks, :]
            test = values[n_train_weeks:, :]
            self.data['test'] = test
            
            # test_target = target.values[n_train_weeks:, :]
            print(train.shape, len(train), test.shape)

            # split into input and outputs
            n_obs = self.n_weeks * self.n_features
            tr_X, tr_y = train[:, :n_obs], train[:, -self.n_features]
            te_X, te_y = test[:, :n_obs], test[:, -self.n_features]
            print(tr_X.shape, len(tr_X), tr_y.shape)

            train_n_X.append(tr_X)
            train_n_y.append(tr_y)
            test_n_X.append(te_X)
            test_n_y.append(te_y)

            # reshape input to be 3D [samples, timesteps, features]
            tr_X = tr_X.reshape((tr_X.shape[0], 26, self.n_features))
            te_X = te_X.reshape((te_X.shape[0], 26, self.n_features))
            print(tr_X.shape, tr_y.shape, te_X.shape, te_y.shape)
            train_X.append(tr_X)
            train_y.append(tr_y)
            test_X.append(te_X)
            test_y.append(te_y)
        
        a =  np.concatenate(train_y, axis=0)
       
        self.data['train_X'] = train_X
        self.data['train_y'] = train_y
        print(a.shape)
        print( len(self.data['train_y']))
        self.data['test_X'] = test_X
        self.data['test_y'] = test_y

        self.data['train_n_X'] = train_n_X
        self.data['train_n_y'] = train_n_y
        self.data['test_n_X'] = test_n_X
        self.data['test_n_y'] = test_n_y

# we have a set of data
# class RSet(RData):


class RModel:

    # Class variable
    # data = None
    # Constructor method

    def __init__(self, data, features, timesteps,  batch_size, n_neurons, n_inputs):
        self.timesteps = timesteps
        self.features = features
        self.batch_size = batch_size
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.lstmInputs = []
        self.lstmLayers = []
        self.data = data
        # self.model = self.load_prev_model()
        self.model = self.create_model()

    def load_prev_model(self):
        # get the data
        data_dir = os.path.join(os.getcwd(), 'models')
        path = data_dir + "/Exp_1-model-LSTM-BB.hdf5"

        # returns a compiled model
        model = load_model(path)
        print('Model loaded')
        return model

    def fit_lstm(self, nb_epoch=1, model_name="model-01-weights.best.hdf5"):
        train_X = self.data['train_X']
        train_y = self.data['train_y']
        test_X = self.data['test_X']
        test_y = self.data['test_y']

        # checkpoint
        models_dir = os.path.join(os.getcwd(), 'models')
        filepath = models_dir + "/" + model_name

        callbacks = [
            EarlyStoppingByLossVal(
                monitor='val_loss', value=0.00001, verbose=1),
            ModelCheckpoint(filepath, monitor='val_loss',
                            save_best_only=True, verbose=0)
            ]
        # fit model

        train_rmse, test_rmse = list(), list()
        for i in range(nb_epoch):
            # train for the different zone
            self.model.fit(
                train_X,
                train_y[0],  # label for the targeted state
                validation_data=(
                    test_X,
                    test_y[0]),
                epochs=1,
                verbose=0,
                shuffle=False,
                batch_size=26,
                callbacks=callbacks
                )
            self.model.reset_states()
        return self.model

    def create_model(self):
        train_X = self.data['train_X']
        train_y = self.data['train_y']

        start = time.time()
        for i in range(self.n_inputs):
            inputName = "{}_input".format(i)

            lstm_input = keras.layers.Input(
                shape=(self.timesteps, self.features),
                batch_shape=(182, 26, self.features),
                name=inputName)
            self.lstmInputs.append(lstm_input)

            lstm_layer = LSTM(self.n_neurons,
                                # (batch size,timesteps,feature shape)
                                batch_input_shape=(182, 26, self.features),
                                stateful=False,
                                return_sequences=True,
                                unroll=True
                                )(self.lstmInputs[i])
            self.lstmLayers.append(lstm_layer)

        # combined the output
        output = keras.layers.concatenate(self.lstmLayers)
        output = Dense(26, activation='relu',
                       name='wheighthedAverage_output')(output)
        stateInput = self.lstmInputs
        model = keras.models.Model(inputs=stateInput, outputs=output)
        start = time.time()
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        print("> Compilation Time : ", time.time() - start)
        model.summary()
        print("Inputs: {}".format(model.input_shape))
        print("Outputs: {}".format(model.output_shape))
        print("Actual input: {}".format(train_X[0].shape))
        print("Actual output: {}".format(train_y[0].shape))
        # save model
        reports_dir = os.path.join(os.getcwd(), 'reports', 'figures')
        d = datetime.datetime.today().strftime("%y-%m-%d")
        directory = os.path.join(reports_dir, 'BB_Model-{}-{}'.format(i, d))
        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath_model = directory + \
            '/BB-lstm_model_{}cells-{}.png'.format(self.n_inputs, d)

        plot_model(model, to_file=filepath_model)
        end = time.time()
        return model

def evaluate(self, model, test_X, test_y):
    scaler = self.data.scaler
    # make a prediction
    yhat = model.predict(test_X)
    #rmse = list()
    for X in test_X : 
        X = X.reshape((X.shape[0], X.shape[2])) #4*51 * self.features
        # invert scaling for forecast
        inv_yhat = numpy.concatenate((yhat, X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = numpy.concatenate((test_y, X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        # calculate RMSE
        # rmse.append(sqrt(mean_squared_error(inv_y, inv_yhat)))
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        #print('RMSE: %.3f' % rmse)
    return rmse

# run a repeated experiment
def experiment(repeats, epochs, param):

    # config
    reports_dir = os.path.join(os.getcwd(), 'reports','figures')
    m_name = "Exp_1-model-LSTM-BB.hdf5"
    d = datetime.datetime.today().strftime("%y-%m-%d")
    directory = os.path.join(reports_dir, 'BB_{}_Exp-{}'.format(epochs,d))
    data = param['data']
    raw = data['raw']
    scaler = data.scaler
    labels = data['cell_labels']
    test_X = data['test_X']
    test_y = data['test_y']
    train_n_X = data['train_n_X']
    test_n_y = data['test_n_y']

    # prepare the data
    model = RModel(**param)
    
    # run experiment
    error_scores = list()
    for r in range(repeats):
        # create a directory for my data
        if not os.path.exists(directory):
            os.makedirs(directory)         
        
        print("REPEATS : {}".format(r))
        # fit the model
        start = time.time()
        lstm_model = model.fit_lstm(epochs, m_name)
        print('> Training Time: {}\n'.format(time.time()-start))
        lstm_model.predict(data['train_X'], batch_size=26)
        
        # forecast test dataset
        start = time.time()
        output = lstm_model.predict(test_X, batch_size=26)
        print("length output :{}".format(len(output)))
        print("length test_y[0] :{}".format(len(test_y[0])))
        print('> Compiling Focast Time: {}\n'.format(time.time()-start))
        
        
        predictions = list()
        for i in range(len(output)):
            yhat = output[i,0]
            print("yhat: {}".format(yhat)) 
			# invert scaling
            yhat = scaler.inverse_transform(yhat)
			# store forecast
            predictions.append(yhat)
        print("length predictions :{}".format(len(predictions)))
        print(predictions)
        # report performance
        rmse = sqrt(mean_squared_error(test_n_y[0], predictions))
        print('> %d) Test RMSE: %.3f' % (r+1, rmse))
        error_scores.append(rmse)

    return error_scores

def load_data():
    # get the data
    data_dir = os.path.join(os.getcwd(), 'data', 'raw')
    path = data_dir + "/2010-2015_ili_climate.csv"
    # longitude + latitude + mean_ili
    data = RData(path)
    return data

def main(): 
    # experiment
    # 3D (sample , 26 , 3)
    data = load_data()

    param = {
        'features': data.n_features,
        'timesteps': 26,
        'batch_size': 26,
        'n_neurons': 1,
        'n_inputs': len(data['raw']),  #I am suppose to have 36
        'data': data
    }
    
    repeats = 1
    results = DataFrame()
    epochs = [1,2,3]

    # vary training epochs
    for e in epochs:
    	results[str(e)] = experiment(repeats, e, param)
    
    # summarize results
    print(results.describe())
    # save boxplot
    results.boxplot()
    pyplot.savefig('boxplot_epochs.png')

if __name__ == '__main__':
    main()
