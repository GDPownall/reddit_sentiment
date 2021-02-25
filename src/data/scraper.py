#!/usr/bin/env python

import praw
import pandas as pd
import numpy as np

read_client_id = {}
with open('client_info.txt','r') as ifile:
    for line in ifile:
        line = line.strip().replace(' ','').split(':')
        read_client_id[line[0]] = line[1]

reddit = praw.Reddit(client_id = read_client_id['client_id'], client_secret=read_client_id['client_secret'], user_agent=read_client_id['user_agent'])


subreddit = reddit.subreddit('AmITheAsshole')

dfs = {}
cats = ['hot','top']
for cat in cats:
    if   cat == 'hot': h = subreddit.hot(limit=int(read_client_id['number_of_posts']))
    elif cat == 'top': h = subreddit.top(limit=int(read_client_id['number_of_posts']))
    else: raise ValueError('Define in script which category to use.')

    posts = []
    n = 0
    for post in h:
        n += 1
        if post.link_flair_text in ['META','Open Forum',None,'None','UPDATE','TL;DR','Update']:continue
        posts.append([
            post.title, post.id, post.selftext, post.created, post.link_flair_text]
            )
    print(n)

    dfs[cat] = pd.DataFrame(posts, columns = ['title','id','body','created','flair'])    
    dfs[cat].set_index('id',inplace=True)

df = pd.concat([dfs[x] for x in dfs.keys()]).drop_duplicates()

for cat in cats:
    df[cat] = df.index.isin(dfs[cat].index).astype(int)

print('Read ',len(df),'posts.')
print (df.head())

try: 
    old = pd.read_pickle('stored_df.pkl')
    new = pd.concat([old,df]).drop_duplicates()
    print('Old dataframe contained',len(old),'posts. Total unique posts:',len(new))
    new.to_pickle('stored_df.pkl')
except OSError:
    df.to_pickle('stored_df.pkl')
