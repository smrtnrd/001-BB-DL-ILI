#!/usr/bin/

# -*- coding: utf-8 -*-
import os
import sys

# add the 'src' directory as one where we can import modules
src_dir = os.path.join(os.getcwd(), os.pardir, 'src')
sys.path.append(src_dir)

import click
import logging

import fnmatch
import glob
from helpers import load_data
from helpers import save_data
from helpers import set_index_to_date

#from dotenv import load_dotenv, find_dotenv

import pandas as pd
from pandas import DataFrame

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    

    # create logger
    logger = logging.getLogger("make_dataset")
    logger.info('START: make_dataset')
    logger.info('===============================================================================')
    logger.info('===============================================================================')
    logger.info('===============================================================================')
    logger.info('===============================================================================')
    logger.info('===============================================================================')
    logger.info('Loading ILI information from the following filepath: %s', input_filepath)
    logger.info('===============================================================================')

    df = load_data(input_filepath)
    logger.info('Select features')
    df = df.groupby(['statename','weekend','ili_activity_label','a_2009_h1n1','week_TEMP','Latitude','Longitude'], as_index=False)['week_TEMP'].mean()
    df = DataFrame(df)
    df = df.iloc[:, :-1] #delete the last col
    df = df.dropna(axis=1)
    #print(df.head())
    df = df.reset_index()
    df = set_index_to_date(df, 'weekend' )
    df = df.reset_index()

    df = df.groupby('statename', as_index=False).apply(lambda x : x.drop_duplicates('date').set_index('date').resample('W').ffill()).reset_index()
    df = df.iloc[:, 1:]
    print(df.head())
    
    

    
    #print(df.head())
    logger.info('DATA SUMMARY')
    logger.info('===============================================================================')
    df.info(verbose=True, null_counts=True)
    logger.info('===============================================================================')
    logger.info("The Data contains {} observations & {}  features".format(df.shape[0], df.shape[1] ))
    logger.info("Missing values: {}".format(df.isnull().values.ravel().sum()))
    logger.info('===============================================================================')


    logger.info("SAVE THE DATA")
    processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    output_filepath = processed_dir + "/001-BB-CDC_ILI_2010-2015_US_STATES-DATA_processed.csv"
    save_data(df, output_filepath)
    logger.info('Data saved in : %s', output_filepath)
    logger.info('===============================================================================')
    logger.info('END: make_dataset')


if __name__ == '__main__':
    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.DEBUG, format=log_fmt)
   
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='test.log',
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

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    
    main()
