#!/usr/bin/

# -*- coding: utf-8 -*-
import os

import click
import logging

import fnmatch
import glob
from helpers import load_data
from helpers import save_data
#from dotenv import load_dotenv, find_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger("make_dataset")
    logger.info('Loading ILI information from the following filepath: %s', input_filepath)
    df = load_data(input_filepath) 
    logger.debug('Records: %s', df)

    processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    output_filepath = processed_dir + "/001-BB-CDC_ILI_2010-2015_US_STATES-DATA_processed.csv"
    logger.info('Data saved in : %s', output_filepath)
    save_data(df, output_filepath)
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    
    main()
