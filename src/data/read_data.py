#!/usr/bin/env python

import pandas as pd
import os

class Data:
    def __init__(self, pkl_path = os.environ['CURRENTDIR']+'/src/data/stored_df.pkl'):
        self.df = pd.read_pickle(pkl_path)

    def one_hot(self):
        one_hot = pd.get_dummies(self.df.flair)
        self.df = self.df.join(one_hot)

    def plot_flairs(self):
        groups = self.df.flair.value_counts()
        print(groups)
        groups.plot.bar()

if __name__ == '__main__':
    x = Data()
    x.plot_flairs()
