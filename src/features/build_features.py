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

    logger.info('DATA SUMMARY')

    logger.info('===============================================================================')
    logger.info("The Data contains {} observations & {}  features".format(df.shape[0], df.shape[1] ))
    logger.info('===============================================================================')

   
    index_name = 'weekend'
    logger.info('2. Converting %s to date format', index_name)
    df = set_index_to_date(df, index_name )
    logger.info('===============================================================================')
    logger.info("Remove Unnamed columns in pandas dataframe")
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    logger.info('===============================================================================')
    
    logger.info('DATA SUMMARY')
    logger.info('===============================================================================')
    df.info(verbose=True, null_counts=True)
    logger.info('===============================================================================')
    logger.info("The Data contains {} observations & {}  features".format(df.shape[0], df.shape[1] ))
    logger.info('===============================================================================')

    logger.info('3. Encode categorical data')
    encoded = encode_cat_data(df)
    logger.info('===============================================================================')
    

    logger.info('DATA SUMMARY')
    logger.info('===============================================================================')
    encoded.info(verbose=True, null_counts=True)

    logger.info('===============================================================================')
    logger.info("The Encoded Data contains {} observations & {}  features".format(encoded.shape[0], encoded.shape[1] ))
    logger.info('===============================================================================')

    logger.info('4. normalize data')
    cname = ['ili_activity_label','a_2009_h1n1','week_TEMP', 'Latitude', 'Longitude']
    norm = norm_data(encoded, cname)
    save_pandas(output_filepath, norm)
    logger.info('5. Data saved in pickle - filepath : %s', output_filepath)
    logger.info('===============================================================================')
    print(norm.head())
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
