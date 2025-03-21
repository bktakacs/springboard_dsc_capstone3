# Prophet Forecasting

import pandas as pd
from prophet import Prophet
import numpy as np
import os
import matplotlib.pyplot as plt

# Load data
datapath = '../data/'
df = pd.read_csv(datapath + '9-8.csv')
df = df[['Average Price of Electricity to Ultimate Customers, Commercial', 'Month']]
df.head()
df.rename(columns={'Average Price of Electricity to Ultimate Customers, Commercial':'y', 'Month':'ds'}, inplace=True)
df.ds = pd.to_datetime(df.ds)
df = df[df.ds > '1990-1-1']

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=2, freq='Y')
m.predict(future)

m.plot(fcst=m.predict(future), include_legend=True)
plt.show()