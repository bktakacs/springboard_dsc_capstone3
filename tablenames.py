# Parse data and extract table description

import numpy as np
import pandas as pd
import os

datapath_data = './springboard_dsc_capstone3/dataex/'
datapath_file = './springboard_dsc_capstone3/'

# Create table-names.txt file to list table names and their descriptions
with open(datapath_file + 'table-names.txt', 'w') as f:    # open txt file  
    for filename in os.listdir(datapath_data): # loop through xlsx files
        if filename.endswith('.xlsx'): # check if file is xlsx
            df = pd.read_excel(
                datapath_data + filename, skiprows=lambda x: x not in [6]
            ) # read only row 7, which contains description
            f.write(str(df.columns[0]) + '\n') # write description


# Create csv's of tables, add tables with incorrect formatting to txt file
with open(datapath_file + 'error-parsing-table.txt', 'w') as f: # write to errorfile
    for filename in os.listdir(datapath_data):
        if filename.endswith('.xlsx'):
            # try: # some files have different time formatting
            df_name = pd.read_excel(datapath_data + filename, skiprows=lambda x: x not in [6]).columns[0].split(sep=' ')[1].replace('.', '-')
            # get name from table description, with - instead of .
            df = pd.read_excel(datapath_data + filename, skiprows=[x for x in range(8)] + [9], header=[0])
            # create df from xlsx file and convert index to datetime
            df.set_index(pd.to_datetime(df.Month, format='%Y %B'), inplace=True)
            df.drop(columns='Month', inplace=True)
            # df.apply(pd.to_numeric, errors='coerce')
            for i in df.columns:
                df[i] = pd.to_numeric(df[i], errors='coerce')  # convert to numeric, replace non-numeric with NaN
            df.to_csv(datapath_data + df_name + '.csv')  # save df as csv
            # except:
            #     print(f'Error parsing {filename}')
            #     f.write(filename + '\n')