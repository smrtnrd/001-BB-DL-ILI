# -*- coding: utf-8 -*-
import os
import click
import logging


import fnmatch
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


@click.command()
@click.option('--norm', is_flag=True)
def main(norm):
    """ Prepare dataset for Deep Learning 
    """

    logger = logging.getLogger("build_features")
    logger.info('select the feature we want to evaluate')
    # read in csv file as a DataFrame
    df = pd.read_csv(input_filepath) 
    df = df[['statename','weekend','ili_activity_label','a_2009_h1n1','week_TEMP','Latitude','Longitude']]
    df.describe() # summary statistics, excluding NaN values
    print(df.weekend.head())
    
    df['weekend'] =  pd.to_datetime(df.weekend)
    print(df.weekend.head())
    df.info(verbose=True, null_counts=True) # concise summary of the table
    
    # set index
    df.set_index('weekend', inplace=True)
    df.index.name = 'date'
   
    #encode categorical features
    df_col = df.columns
    col_non_num = [c for c in df_col if df[c].dtype == 'object']
    for c in col_non_num:
        df[c] = LabelEncoder().fit_transform(df[c])


    ## Transform data
    #normalize data 
    logger = logging.getLogger(__name__)
    logger.info('normalize data')
    
    if norm:        
        scaler = MinMaxScaler()
        df['ili_activity_label'] = scaler.fit_transform(df['ili_activity_label'].values.reshape(-1,1))
        df['a_2009_h1n1'] = scaler.fit_transform(df['a_2009_h1n1'].values.reshape(-1,1))
        df['week_TEMP'] = scaler.fit_transform(df['week_TEMP'].values.reshape(-1,1))
        df['Latitude'] = scaler.fit_transform(df['Latitude'].values.reshape(-1,1))
        df['Longitude'] = scaler.fit_transform(df['Longitude'].values.reshape(-1,1))
    
    df.to_csv(output_filepath)
    df.info()    

    logger = logging.getLogger(__name__)
    logger.info('transform processed data set from from processed to interm data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
