import numpy as np

import pandas as pd
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
       Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    print("The data contains {} features".format(n_vars))
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    n_vars = len(agg.columns)
    print("The reframed data contains {} features".format(n_vars))

    if dropnan:
        agg.dropna(inplace=True)
        return agg

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
	new_row = [x for x in X] + [yhat]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]


# One Hot encoder
def encode_category(data):
    #check the categories
    df = pd.get_dummies(data)
    return df

# parse the data into states
def parse_data(states_label, datasets, n_weeks = 260):
    """
    return a dictionary
    """
    parse_data = {}
    for state in states_label:
        data = datasets[state]
        names = list(data.columns.values)
        
        #drop date value
        if 'date' in names:
            data.drop(['date'], 1, inplace=True)

        # make sure that every states has the same number of weeks
        if(len(data)>= n_weeks):
            parse_data[state] = data
    return parse_data


def reframe_data(data_states, n_weeks =1,n_features =1 ):
    reframed_data = []
    for state in data_states:
        values=data_states[state]
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        reframed = series_to_supervised(values, n_weeks, n_features)
        # we are predicting ili activity
        reframed_data.append(reframed)
    return reframed_data


