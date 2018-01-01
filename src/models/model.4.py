import os, sys, errno
import warnings

from math import sqrt
import numpy

import pydot
import graphviz

# Take a look at the raw data :
import pandas as pd
from pandas import DataFrame
from pandas import read_csv

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import matplotlib
# be able to save images on server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


import keras
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
# be able to save images on server
# matplotlib.use('Agg')
import time
import datetime

import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings


class RData:
    def __init__(self, path, n_weeks=1):
        self.path = path
        self.data = {}
        # add raw database
        self.data['raw'] = self.load_data()
        # scale data
        self.scaler = preprocessing.MinMaxScaler()
        self.scale()
        # reframe data
        self.reframe()
        # self.state_list_name = self.data.state.unique()
        self.n_weeks = n_weeks
        self.n_features = int(len(self.data['raw'][0].columns))
        print("number of features: {}".format(self.n_features))
        
        self.split_data()
        #print(self.n_features)

    # Return specific data
    def __getitem__(self, index):
        return self.data[index]

    # convert series to supervised learning
    @staticmethod
    def series_to_supervised(data, n_in=26, n_out=26, dropnan=True):
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
        agg.columns = names
        
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
        
    def load_data(self):
        raw = read_csv(self.path)
        raw = raw.fillna(0)
        # print(raw['0'].head())
        #raw = raw.drop(["0"],  axis = 1)
        #print(raw.head())

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
                        'latitude',
                        'longitude',
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

    # normalize
    def scale(self):
        scaled = list()
        for df in self.data['raw']:
            scaled_df = self.scaler.fit_transform(df)
            scaled_df = pd.DataFrame(scaled_df, columns=df.columns.values)
            scaled.append(scaled_df)
        self.data['scaled'] = scaled

    def reframe(self, n_weeks=26):
        # specify the number of lag_weeks
        reframed = list()
        for df in self.data['scaled']:
            # frame as supervised learning
            reframed.append(self.series_to_supervised(df, n_weeks))
        self.data['reframed'] = reframed

    # Return specific data
    def split_data(self):
        # split into train and test sets
        train_X, train_y = list(), list()
        test_X, test_y = list(), list()

        for reframed in self.data['reframed']:
            values = reframed.values
            n_train_weeks = 52 * 4
            train = values[:n_train_weeks, :]
            test = values[n_train_weeks:, :]
            # split into input and outputs
            n_obs = self.n_weeks * self.n_features
            tr_X, tr_y = train[:, :n_obs], train[:, -self.n_features]
            te_X, te_y = test[:, :n_obs], test[:, -self.n_features]
            #print(tr_X.shape, len(tr_X), tr_y.shape)
            # reshape input to be 3D [samples, timesteps, features]
            tr_X = tr_X.reshape((tr_X.shape[0], self.n_weeks, self.n_features))
            te_X = te_X.reshape((te_X.shape[0], self.n_weeks, self.n_features))
            #print(tr_X.shape, tr_y.shape, te_X.shape, te_y.shape)
            train_X.append(tr_X)
            train_y.append(tr_y)
            test_X.append(te_X)
            test_y.append(te_y)
        self.data['train_X'] = train_X
        self.data['train_y'] = train_y
        self.data['test_X'] = test_X
        self.data['test_y'] = test_y


# we have a set of data
# class RSet(RData):


class RModel:

    # Class variable
    # data = None
    # Constructor method

    def __init__(self, data, features, timesteps,  batch_size, n_neurons, n_inputs ):
        self.timesteps = timesteps
        self.features = features
        self.batch_size = batch_size
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.lstmInputs = []
        self.lstmLayers = []
        self.data = data
        self.model = self.create_model() 

    def fit_lstm(self, nb_epoch=1, model_name="model-01-weights.best.hdf5"):
        train_X = self.data['train_X']
        train_y = self.data['train_y']
        test_X = self.data['test_X']
        test_y = self.data['test_y']

        # checkpoint
        models_dir = os.path.join(os.getcwd(), 'models')
        filepath= models_dir + "/" + model_name
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        # fit model
        train_rmse, test_rmse = list(), list()
        for i in range(nb_epoch):
            for a in range(len(self.data['raw'])): #train for the different zone
                self.model.fit(
                    train_X,
                    train_y[a],  # label for the targeted state
                    validation_data=(
                        test_X,
                        test_y[a]),
                    epochs=1,
                    verbose=0,
                    shuffle=False,
                    batch_size=self.batch_size,
                    callbacks=callbacks_list
                    )
                self.model.reset_states()
        return self.model
        


    def create_model(self):
        start = time.time()
        for i in range(self.n_inputs):
            inputName = "{}_input".format(i)

            lstm_input = keras.layers.Input(
                shape=(self.timesteps, self.features),
                name=inputName)
            self.lstmInputs.append(lstm_input)

            lstm_layer = LSTM(self.n_neurons,
                              return_sequences=False)(self.lstmInputs[i])
            self.lstmLayers.append(lstm_layer)

        # combined the output
        output = keras.layers.concatenate(self.lstmLayers)
        output = Dense(1, activation='relu',
                       name='wheighthedAverage_output')(output)
        stateInput = self.lstmInputs
        model = keras.models.Model(inputs=stateInput, outputs=[output])
        start = time.time()
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        print("> Compilation Time : ", time.time() - start)

        #save model
        reports_dir = os.path.join(os.getcwd(), 'reports','figures')
        d = datetime.datetime.today().strftime("%y-%m-%d")
        directory = os.path.join(reports_dir, 'BB_Model-{}-{}'.format(i,d))
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        filepath_model  = directory + '/BB-lstm_model_{}cells-{}.png'.format(self.n_inputs, d)
        plot_model(model, to_file=filepath_model)
        end = time.time()
        return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def plot_RMSE(history, title):
    plt.plot(history['train'], color='orange')
    plt.plot(history['test'], color='green')
    plt.title(title)
    plt.ylabel('RMSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
# Model Evaluation - RMSE
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def predict_default(model, scaler, test_X, test_y, batch_size = 52):
    yhat = model.predict(test_X, batch_size=batch_size)
    raw_values = list()
    predictions = list()
    for X in test_X : 
        for Y in test_y:
            print(X)            
            X = X.reshape((X.shape[0], X.shape[2])) #4*51 * self.features
            # make a prediction
            
            # invert scaling for forecast
            inv_yhat = numpy.concatenate((yhat, X[:, 1:]), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:,0]
            print(inv_yhat)
            # invert scaling for actual
            print(Y)
            Y = Y.reshape((len(Y), 1))
            print(Y)
            inv_y = numpy.concatenate((Y, X[:, 1:]), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            inv_y = inv_y[:,0]
            raw_values.append(inv_y)
            predictions.append(yhat)
            
            # calculate RMSE
            # rmse.append(sqrt(mean_squared_error(inv_y, inv_yhat)))
    
    return raw_values, predictions


def evaluate(raw_values, predictions):
    raw_values = list()
    predictions = list()
    print("raw_values")
    print(len(raw_values))
    print("predictions")
    print(len(predictions)) 
    print(predictions)  
    rmse = sqrt(mean_squared_error(raw_values, predictions))
    return rmse

# run a repeated experiment
def experiment(repeats, epochs, param):

    # config
    reports_dir = os.path.join(os.getcwd(), 'reports','figures')
    m_name = "Exp_1-model-LSTM-BB.hdf5"
    d = datetime.datetime.today().strftime("%y-%m-%d")
    directory = os.path.join(reports_dir, 'BB_{}_Exp-{}'.format(epochs,d))
    data = param['data']
    labels = data['cell_labels']
    test_X = data['test_X']
    test_y = data['test_y']

    #prepare the data
    model = RModel(**param)
    
    # run experiment
    error_scores = list()
    for r in range(repeats):
        #create a directory for my data
        if not os.path.exists(directory):
            os.makedirs(directory)         
        
        print("REPEATS : {}".format(r))
        #for m in range(param['n_inputs']):
        start = time.time()
        lstm_model = model.fit_lstm(epochs, m_name)
        print('> Training Time: {}\n'.format(time.time()-start))
        # forecast test dataset
        raw_values, predictions = predict_default(lstm_model, data.scaler, test_X, test_y)
        #title = 'Model loss for {}'.format(epochs)
        #plot history data
        #for index,history in enumerate(rhistory):   
        #    plot_RMSE(history, title)

        #filepath  = directory + '/{}_exp_{}_epochs_rmse_2010-2014.png'.format(r, epochs)
        #print('data stored: {}'.format(filepath))
        #plt.savefig(filepath)
        #plt.close()
        rmse = evaluate(raw_values, predictions)
        print('%d) Test RMSE: %.3f' % (r+1, rmse))
        error_scores.append(rmse)

    return error_scores

def load_data():
    # get the data
    data_dir = os.path.join(os.getcwd(), 'reports','figures')
    path = "/Users/bbuildman/Documents/Developer/GitHub/001-BB-DL-ILI/data/raw/2010-2015_ili_climate.csv"
    #longitude + latitude + mean_ili
    data = RData(path)
    return data

def main(): 
    # experiment
    data = load_data()

    param = {
        'features': data.n_features,
        'timesteps': 1,
        'batch_size': 52,
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