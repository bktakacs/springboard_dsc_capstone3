import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss

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

def arima_mse(data, order):
    split = int(len(data) * 0.8)
    train, test = data[0:split], data[split:]
    past = [x for x in train]

    predictions = list()
    for i in range(len(test)):
        model = ARIMA(past, order=order)
        model_fit = model.fit()
        future = model_fit.forecast()[0]
        predictions.append(future)
        past.append(test[i])

    error = mean_squared_error(test, predictions)
    
    return error

################################################################################

def model_eval(data, pvals, dvals, qvals):
    '''
    Function to evaluate different ARIMA models with several different p, d, and q values.
    '''
    best_score, best_cfg = float('inf'), None
    for p in pvals:
        for d in dvals:
            for q in qvals:
                order = (p, d, q)
                try:
                    mse = arima_mse(data, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3E' % (order, mse))
                    
                except:
                    continue
    print('Best ARIMA%s MSE=%.3E' % (best_cfg, best_score))
    return best_cfg

################################################################################

def decomposition(data, model='additive'):
    decomp = seasonal_decompose(data, model=model)

    trend = decomp.trend
    seasonal = decomp.seasonal
    residual = decomp.resid

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