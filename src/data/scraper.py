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
hot = subreddit.top(limit=int(read_client_id['number_of_posts']))

posts = []
n = 0
for post in hot:
    n += 1
    if post.link_flair_text in ['META','Open Forum',None,'None','UPDATE','TL;DR','Update']:continue
    posts.append([
        post.title, post.id, post.selftext, post.created, post.link_flair_text]
        )
print(n)

df = pd.DataFrame(posts, columns = ['title','id','body','created','flair'])

df.set_index('id',inplace=True)
print (df.head())

df.to_pickle('stored_df.pkl')