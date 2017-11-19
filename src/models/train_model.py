# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import click
import logging
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from pandas import DataFrame
from pandas import concat

from numpy import concatenate
from matplotlib import pyplot

from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
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
	if dropnan:
		agg.dropna(inplace=True)
	return agg

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
#@click.option('--n', nargs=1, type=float, help='Prop of training set')
def main(input_filepath):
    """ Prepare dataset for Deep Learning 
    """

    logger = logging.getLogger(__name__)
    logger.info('#######################################')
    logger.info('Train Model')
    # read in csv file as a DataFrame
    df = pd.read_csv(input_filepath)

    df['date'] =  pd.to_datetime(df.date)
    print(df.date.head())
    df.info(verbose=True, null_counts=True) # concise summary of the table

    print(df.date.head())

    # set index
    df.set_index('date', inplace=True)
    
    logger = logging.getLogger(__name__)
    logger.info('Peepare data form time series analysis')    

    # specify the number of lag hours
    n_weeks = 4
    n_features = 8

    reframed = series_to_supervised(df, n_weeks, 4)   
    
    # split into train and test sets
    values = reframed.values
    n_train_weeks = 4 * 12
    train = values[:n_train_weeks, :]
    test = values[n_train_weeks:, :]


    #click.echo(reframed.head())
    # split into input and outputs
    n_obs = n_weeks * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_weeks, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_weeks, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    logger = logging.getLogger(__name__)
    logger.info('transform processed data set from from processed to interm data')

    # design network
    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=16, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
 


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    
    main()
