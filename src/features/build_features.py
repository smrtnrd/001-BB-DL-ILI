#!/usr/bin/

# -*- coding: utf-8 -*-
import os
import logging

import fnmatch
import glob

from helpers import load_data
from helpers import save_data
from helpers import norm_data
from helpers import encode_cat_data
from helpers import set_index_to_date
from helpers import save_pandas
from helpers import plot_data
from helpers import MultiColumnLabelEncoder

from datetime import datetime
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def main():
    """ Prepare dataset for Deep Learning 
    """

    logger = logging.getLogger("build_features")
    logger.info('START: build_features')
    logger.info('===============================================================================')
    logger.info('===============================================================================')
    logger.info('===============================================================================')
    logger.info('===============================================================================')
    logger.info('===============================================================================')
    
    processed_dir = os.path.join(os.getcwd(), 'data')
    input_filepath = processed_dir + "/processed/001-BB-CDC_ILI_2010-2015_US_STATES-DATA_processed.csv"
    output_filepath = processed_dir + "/interim/001-BB-CDC_ILI_2010-2015_US_STATES-DATA_interim.pickle"

    logger.info('1. Loading ILI information from the following filepath: %s', input_filepath)
    df = load_data(input_filepath)
    df = df.dropna(how='any') 
    logger.info('DATA SUMMARY')

    logger.info('===============================================================================')
    logger.info("The Data contains {} observations & {}  features".format(df.shape[0], df.shape[1] ))
    logger.info('===============================================================================')

   
    index_name = 'date'
    logger.info('2. Converting %s to date format', index_name)
    df = set_index_to_date(df, index_name )
    logger.info('===============================================================================')
    logger.info("Remove Unnamed columns in pandas dataframe")
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.loc[:, ~df.columns.str.contains('^level_0')]
    logger.info('===============================================================================')
    print(df.groupby(level=0).count().head())
    print(df.mode().head())
    df = MultiColumnLabelEncoder(columns = ['ili_activity_label']).fit_transform(df)
    df = df.groupby('statename', as_index=False).resample('W').ffill().reset_index()
    df = df.iloc[:, 1:]
    df = set_index_to_date(df, 'date' )

    #print(df.head())
    #print(df.tail())

    #print(df.groupby(level=0).count().head())
    #print(df.groupby(level=0).count())
    logger.info('DATA SUMMARY')
    logger.info('===============================================================================')
    df.info(verbose=True, null_counts=True)
    logger.info('===============================================================================')
    logger.info("The Data contains {} observations & {}  features".format(df.shape[0], df.shape[1] ))
    logger.info('===============================================================================')
    
   
    logger.info('3. Encode categorical data')
    encoded =  MultiColumnLabelEncoder(columns = ['statename']).fit_transform(df)
    logger.info('===============================================================================')
    

    logger.info('DATA SUMMARY')
    logger.info('===============================================================================')

    encoded.dropna(inplace=True)
    
    encoded.info(verbose=True, null_counts=True)
    logger.info('===============================================================================')
    logger.info("The Encoded Data contains {} observations & {}  features".format(encoded.shape[0], encoded.shape[1] ))
    logger.info("Missing values: {}".format(encoded.isnull().values.ravel().sum()))
    logger.info('===============================================================================')

    logger.info('4. normalize data')
    cname = ['ili_activity_label','a_2009_h1n1','week_TEMP']
    norm = norm_data(encoded, cname)
    
    print(norm.groupby(level=0).count().head())
    save_pandas(output_filepath, norm)
    logger.info('5. Data saved in pickle - filepath : %s', output_filepath)
    logger.info('===============================================================================')
    print(norm.head())
    plot_data(norm,'a_2009_h1n1')

    logger.info('===============================================================================')
    logger.info('END: build_features')
         
if __name__ == '__main__':
     # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='build_features.log',
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
