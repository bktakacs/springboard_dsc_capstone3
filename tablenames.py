# Parse data and extract table description

import numpy as np
import pandas as pd
import os

datapath = './springboard_dsc_capstone3/data/'

# Create table-names.txt file to list table names and their descriptions
with open(datapath + 'table-names.txt', 'w') as f:    # open txt file  
    for filename in os.listdir(datapath): # loop through xlsx files
        if filename.endswith('.xlsx'): # check if file is xlsx
            df = pd.read_excel(
                datapath + filename, skiprows=lambda x: x not in [6]
            ) # read only row 7, which contains description
            f.write(str(df.columns[0]) + '\n') # write description


# Create csv's of tables, add tables with incorrect formatting to txt file
with open(datapath + 'error-parsing-table.txt', 'w') as f: # write to errorfile
    for filename in os.listdir(datapath):
        if filename.endswith('.xlsx'):
            try: # some files have different time formatting
                df_name = pd.read_excel(datapath + filename, skiprows=lambda x: x not in [6]).columns[0].split(sep=' ')[1].replace('.', '-')
                # get name from table description, with - instead of .
                df = pd.read_excel(datapath + filename, skiprows=[x for x in range(7)] + [9], header=[1], index_col=0)
                # create df from xlsx file and convert index to datetime
                df.index = pd.to_datetime(df.index, format='%Y %B')
                df.to_csv(datapath + df_name + '.csv')  # save df as csv file
            except:
                print(f'Error parsing {filename}')
                f.write(filename + '\n')