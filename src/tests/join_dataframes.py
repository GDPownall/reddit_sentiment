#!/usr/bin/env python

'''
A script for testing the joining of two dataframes while storing which dataframe each index appeared in
'''

import pandas as pd
import numpy as np


df1 = pd.DataFrame(
        { 'id': [1,2,3,4,6],
          'A' : [1,2,3,4,6],
          'B' : [5,6,7,8,10],
          }
        )

df1.set_index('id',inplace=True)

df2 = pd.DataFrame(
        { 'id': [1,2,3,4,5],
          'A' : [1,2,3,4,5],
          'B' : [5,6,7,8,9],
          }
        )

df2.set_index('id',inplace=True)

df3 = pd.concat([df1,df2]).drop_duplicates()

df3['1'] = df3.index.isin(df1.index).astype(int)
df3['2'] = df3.index.isin(df2.index).astype(int)


print(df1)
print(df2)
print(df3)
