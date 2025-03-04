import numpy as np
import pandas as pd
import os

datapath = './data/'

# Function to load and preprocess data
def load_table_data(filename):
    try:
        df = pd.read_csv(datapath + filename)
        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print('Data loaded from: {}'.format(filename))
    
    except:
        print('Error loading data from: {}'.format(filename))
        df = None

    return df

def load_csv_data(filename):
    try:
        df = pd.read_csv(datapath + filename)
        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)
        print('Data loaded from: {}'.format(filename))
    
    except:
        print('Error loading data from: {}'.format(filename))
        df = None

    return df