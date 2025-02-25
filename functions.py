import numpy as np
import pandas as pd
import os

datapath = './data/'

# Function to load and preprocess data
def load_data(filename, verbose: bool = False):
    df = pd.read_csv(datapath + filename)
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if verbose:
        print(f'Data loaded from: {filename}', df.isnull().sum())