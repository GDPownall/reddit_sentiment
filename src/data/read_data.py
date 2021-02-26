#!/usr/bin/env python

import pandas as pd
import os

class Data:
    def __init__(self, pkl_path = None):
        if pkl_path == None:
            try: WKDIR = os.environ['CURRENTDIR']
            except KeyError: 
                print('CURRENTDIR not defined. Remember to do source setup.sh.')
                raise
            pkl_path = WKDIR+'/src/data/stored_df.pkl'
        self.df = pd.read_pickle(pkl_path)
        self.one_hot_encoded = False

    def one_hot(self):
        if self.one_hot_encoded:
            pass
        else:
            one_hot = pd.get_dummies(self.df.flair)
            self.df = self.df.join(one_hot)
            self.one_hot_encoded = True

    from plotting_fcns import plot_flairs, flairs_over_time

if __name__ == '__main__':
    x = Data()
    x.plot_flairs('x.pdf')
    x.one_hot()
    x.flairs_over_time()
