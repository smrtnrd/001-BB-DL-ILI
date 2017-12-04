from __future__ import print_function


import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from pandas import read_csv
import cPickle as pickle

from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
import seaborn as sns
sns.set()

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import os
import logging

from os import listdir


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


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
    #print(df.head())
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

def plot_data(df, cname):
    processed_dir = os.path.join(os.getcwd(), 'reports', 'figures')
    fig3 = pyplot.figure()
    result = seasonal_decompose(df[cname], model='additive', freq=1)
    result.plot()
    pyplot.show()
    #output_filepath = processed_dir + "/Additive_Model_Decomposition_Plot.png"
    #print(output_filepath)
    #fig3.savefig(output_filepath)

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
        numpy.lib.format.read_magic(f)
        numpy.lib.format.read_array_header_1_0(f)
        f.seek(values.dtype.alignment * values.size, 1)
        meta = pickle.loads(f.readline().decode('string_escape'))
    if len(meta) == 2:
        return pd.DataFrame(values, index=meta[0], columns=meta[1])
    elif len(meta) == 1:
        return pd.Series(values, index=meta[0])

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)