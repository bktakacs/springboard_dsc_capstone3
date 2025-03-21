import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss

datapath = '../data/'

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
        print(os.getcwd())
        df = None

    return df

################################################################################

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

################################################################################

def decomposition(data, model='additive'):
    decomp = seasonal_decompose(data, model=model)

    trend = decomp.trend
    seasonal = decomp.seasonal
    residual = decomp.resid

    plt.figure(figsize=(12,8))
    plt.subplot(411)
    plt.plot(data, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    return decomp

################################################################################
# def plot_model(fitted_model, forecast, title, ):
