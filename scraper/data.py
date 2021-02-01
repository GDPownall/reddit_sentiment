#!/usr/bin/env python

import pandas as pd

df = pd.read_pickle('stored_df.pkl')
one_hot = pd.get_dummies(df['flair'])
df = df.join(one_hot)
print(len(df))
print(df.columns)
