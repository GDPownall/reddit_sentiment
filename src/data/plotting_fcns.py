#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_flairs(self,saveas = None):
    groups = self.df.flair.value_counts()
    groups.plot(kind='barh')
    plt.subplots_adjust(left=0.2)
    if saveas == None: plt.show()
    else: plt.savefig(saveas)

def flairs_over_time(self):
    gap = 60*60*24*7
    print(self.df.created.head(20))
    bins = np.arange(self.df['created'].min(), self.df['created'].max(), gap)
    new_df = self.df.groupby(pd.cut(self.df['created'], np.arange(self.df['created'].min(), self.df['created'].max(), gap))).sum()
    flairs = list(self.df.flair.unique())
    new_df = new_df[flairs]

    new_df.plot.area()
    plt.show()
    return
