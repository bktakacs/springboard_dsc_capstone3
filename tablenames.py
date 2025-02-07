# Parse data and extract table description

import numpy as np
import pandas as pd
import os

datapath = './springboard_dsc_capstone3/data/'

with open(datapath + 'table-names.txt', 'w') as f:     
    for filename in os.listdir(datapath):
        if filename.endswith('.xlsx'):
            df = pd.read_excel(
                datapath + filename, skiprows=lambda x: x not in [6]
            )
            f.write(str(df.columns[0]) + '\n')