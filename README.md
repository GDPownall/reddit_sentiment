# reddit_scraper

Attempt to use sentiment-analysis machine learning to predict the verdict of the AmITheAsshole subreddit.
This is a very nice set of data, with crowd-sourced sentiment. However, any models produced probably can't be used anywhere else.

Packages:
pandas
praw

## Scraping the data

This is set up in the scraper/ directory.
First, you need to make a client_info.txt file with the following praw package settings 

```
client_id:your_client_id
client_secret:your_client_secret
user_agent:your_user_agent
number_of_posts:number_of_posts_to_scrape
```

Then enter the scraper/ directory and run

```bash
python3 scraper.py
```

...which will store the scraped data as a dataframe. The outcome of each post is stored by one-hot encoding. 
