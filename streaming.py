#%%
import sys

import numpy as np

import tweepy

from utils import (
    blob_clean_tweets,
    vader_clean_tweets,
)

import settings

consumer_key = settings.consumer_key
consumer_secret = settings.consumer_secret
access_key = settings.access_key
access_secret = settings.access_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


def getText(status):
    if hasattr(status, "retweeted_status"):  # Check if Retweet
        try:
            text = status.retweeted_status.extended_tweet["full_text"]
        except AttributeError:
            text = status.retweeted_status.text
    else:
        try:
            text = status.extended_tweet["full_text"]
        except AttributeError:
            text = status.text

    return text


class SentimentStreamListener(tweepy.StreamListener):

    def __init__(self):
        super().__init__()
        self.analyser = SentimentIntensityAnalyzer()

    def on_status(self, status):

        data = {
            status.created_at,
            status.user,
            status.source,
            status.favorite_count,
            status.retweet_count,
            status.reply_count
        }

        raw_text = getText(status)

        # VADER
        text = vader_clean_tweets(raw_text)
        
        scores = self.analyser.polarity_scores(text)
        data += [scores['neg'], scores['neu'], scores['pos'], scores['compound']]

        # TextBlob
        blob_text, blob = blob_clean_tweets(raw_text)
        #pass textBlob method for sentiment calculations
        Sentiment = blob.sentiment

        #seperate polarity and subjectivity in to two variables
        polarity = Sentiment.polarity
        subjectivity = Sentiment.subjectivity

        data += [polarity, subjectivity]

        #print(data)


    def on_error(self, status_code):
        print('Encountered error with status code:', status_code)
        return True # Don't kill the stream

    def on_timeout(self):
        print('Timeout...')
        return True # Don't kill the stream


# %%
if __name__ == "__main__":
    
    sapi = tweepy.streaming.Stream(auth, SentimentStreamListener())
    sapi.filter(track=['bitcoin'], languages=['en'])


# %%
