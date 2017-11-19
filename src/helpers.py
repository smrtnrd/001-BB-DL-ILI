import numpy as np
import pandas as pd
from pandas import read_csv


def load_data(input_filepath):
    df = read_csv(input_filepath)
    return df

def save_data(df, output_filepath):
    df.to_csv(output_filepath)