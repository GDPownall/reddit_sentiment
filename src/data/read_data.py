#!/usr/bin/env python

import pandas as pd
import os
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch

RANDOM_SEED = 42
labels = ['Asshole','Not the A-hole']
labels += ['Everyone Sucks', 'No A-holes here']

class Data:
    def __init__(self, df, pre_trained_model_name = 'bert-base-cased', max_len = 100 ):
        self.df = df[df['flair'].isin(labels)]
        self.df['body'] = self.df['body'].apply(lambda x: ' '.join(x.split()[:600]))
        self.one_hot_encoded = False
        self.split_data = None

        self.pre_trained_model_name = pre_trained_model_name
        self.tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
        self.max_len = max_len

    @classmethod
    def from_pkl(cls, pkl_path = None, pre_trained_model_name = 'bert-base-cased', max_len = 100 ): 
        if pkl_path == None:
            try: WKDIR = os.environ['CURRENTDIR']
            except KeyError:
                print('CURRENTDIR not defined. Remember to do source setup.sh.')
                raise
            pkl_path = WKDIR+'/src/data/stored_df.pkl'
        df = pd.read_pickle(pkl_path) # columns ['title', 'body', 'created', 'flair', 'hot', 'top']
        return cls(df, pre_trained_model_name = pre_trained_model_name, max_len = max_len)

    def __len__(self):
        return len(self.df)

    def __getitem__(self,item):
        title = list(self.df.title[item])
        text  = list(self.df.body[item])
        flair = list(self.df.flair[item])

        encoding = self.tokenizer.batch_encode_plus(
                    text,
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    padding = True,
                    max_length = self.max_len,
                    truncation = True, 
                    return_attention_mask=True,
                    return_tensors='pt' #pytorch
                ) 

        try: target = self.df[labels][item].to_numpy() 
        except KeyError: 
            print('Have you remembered to one-hot encode the data? Data.one_hot()')
            raise
        
        #input_ids = torch.Tensor(len(encoding['input_ids']), 

        return {
                'title':title,
                'text_body':text,
                'flair': flair,
                'input_ids':encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'targets': target
                }

    def one_hot_weights(self):
        try: target = self.df[labels].to_numpy()
        except KeyError:
            print('Have you remembered to one-hot encode the data? Data.one_hot()')
            raise

        weights = target.sum(axis=0)
        return 1/weights
 

    def one_hot(self):
        if self.one_hot_encoded:
            pass
        else:
            one_hot = pd.get_dummies(self.df.flair)
            self.df = self.df.join(one_hot)
            self.one_hot_encoded = True

    def n_classes(self):
        return 4

    def clone_new_df(self, new_df):
        '''
        return clone with new dataframe
        '''
        new = Data(
                new_df, 
                pre_trained_model_name = self.pre_trained_model_name,
                )
        new.one_hot_encoded = self.one_hot_encoded
        return new

    def train_test_split(self):
        if self.split_data != None:
            return 
        print('Splitting into train and test sets.')
        # for stratifying
        label_dict = {}
        for idx, label in enumerate(labels):
            label_dict[label] = idx
        strat = self.df.flair.replace(label_dict).values

        df_train, df_test = train_test_split(
                self.df,
                test_size = 0.1,
                random_state = RANDOM_SEED,
                stratify = strat
                )
        
        strat2 = df_test.flair.replace(label_dict).values
        df_val, df_test = train_test_split(
                df_test,
                test_size = 0.5,
                random_state = RANDOM_SEED,
                stratify = strat2
                )
        self.split_data = (
                self.clone_new_df(df_train), 
                self.clone_new_df(df_val), 
                self.clone_new_df(df_test) 
                )

    from data.plotting_fcns import plot_flairs, flairs_over_time

if __name__ == '__main__':
    x = Data.from_pkl()
    x.plot_flairs('x.pdf')
    x.one_hot()
    print(x.df.columns)
    x.flairs_over_time()
