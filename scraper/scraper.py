#!/usr/bin/env python

import praw
import pandas as pd

read_client_id = {}
with open('client_info.txt','r') as ifile:
    for line in ifile:
        line = line.strip().replace(' ','').split(':')
        read_client_id[line[0]] = line[1]

reddit = praw.Reddit(client_id = read_client_id['client_id'], client_secret=read_client_id['client_secret'], user_agent=read_client_id['user_agent'])


subreddit = reddit.subreddit('AmITheAsshole')
hot = subreddit.hot(limit=100000)

posts = []
for post in hot:
    if post.link_flair_text in ['META','Open Forum',None,'None','UPDATE','TL;DR']:continue
    posts.append([
        post.title, post.id, post.selftext, post.created, post.link_flair_text]
        )


df = pd.DataFrame(posts, columns = ['title','id','body','created','flair'])

df.set_index('id',inplace=True)
one_hot = pd.get_dummies(df['flair'])
df = df.join(one_hot)
print (df.head())

df.to_pickle('stored_df.pkl')
