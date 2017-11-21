from __future__ import print_function

import pandas as pd
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from pandas import read_csv
import cPickle as pickle

import os

import logging


def encode_cat_data(data):
    logging.debug("Processing value '%s'", data)
    """
        Take a dataframe as input
        encode categorical features
    """
    df = data
    df_col = df.columns
    col_non_num = [c for c in df_col if df[c].dtype ==
                   'object']  # fetch all non num data
    for c in col_non_num:
        df[c] = LabelEncoder().fit_transform(df[c])
    return df


def set_index_to_date(data, cname):
    """
        transform index to the right date
    """
    df = data
    df[cname] = pd.to_datetime(df[cname])
    # set index
    df.set_index(cname, inplace=True)
    df.index.name = 'date'
    print(df.head())
    return df


def load_data(input_filepath):
    """
    Load data from csv
    
    """
    df = read_csv(input_filepath)
    return df


def save_data(df, output_filepath):
    """
    save data

    """
    df.to_csv(output_filepath)


def norm_data(df, raw_data_list):

    scaler = MinMaxScaler()
    for col in raw_data_list:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    return df


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


def save_pandas(fname, data):
    '''Save DataFrame or Series
    Parameters
    ----------
    fname : str
        filename to use
    data: Pandas DataFrame or Series
    '''
    np.save(open(fname, 'w'), data)
    if len(data.shape) == 2:
        meta = data.index, data.columns
    elif len(data.shape) == 1:
        meta = (data.index,)
    else:
        raise ValueError('save_pandas: Cannot save this type')
    s = pickle.dumps(meta)
    s = s.encode('string_escape')
    with open(fname, 'a') as f:
        f.seek(0, 2)
        f.write(s)


def load_pandas(fname, mmap_mode='r'):
    '''Load DataFrame or Series
    Parameters
    ----------
    fname : str
        filename
    mmap_mode : str, optional
        Same as numpy.load option
    '''
    values = np.load(fname, mmap_mode=mmap_mode)
    with open(fname) as f:
        np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f)
        f.seek(values.dtype.alignment * values.size, 1)
        meta = pickle.loads(f.readline().decode('string_escape'))
    if len(meta) == 2:
        return pd.DataFrame(values, index=meta[0], columns=meta[1])
    elif len(meta) == 1:
        return pd.Series(values, index=meta[0])


