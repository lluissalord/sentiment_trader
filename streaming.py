#%%
import sys

import numpy as np

import tweepy

from utils import (
    blob_clean_tweets,
    vader_clean_tweets,
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Generalised function to extract text from tweets
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
    """
    Listener extracting sentiment while streaming all tweets defined by filter
    """
    def __init__(self):
        super().__init__()
        self.analyser = SentimentIntensityAnalyzer()

    def on_status(self, status):

        # Data that will be stored directly from twitter
        data = [
            status.created_at,
            #status.user,
            status.source,
            status.favorite_count,
            status.retweet_count,
            status.reply_count
        ]

        # Extract text from tweet
        raw_text = getText(status)

        # VADER process
        text = vader_clean_tweets(raw_text)
        
        # Compute VADER sentiment on text
        scores = self.analyser.polarity_scores(text)
        data += [scores['neg'], scores['neu'], scores['pos'], scores['compound']]

        # TextBlob process
        blob_text, blob = blob_clean_tweets(raw_text)

        # Pass textBlob method for sentiment calculations
        Sentiment = blob.sentiment

        # Seperate polarity and subjectivity in to two variables
        polarity = Sentiment.polarity
        subjectivity = Sentiment.subjectivity
        data += [polarity, subjectivity]

        # TODO: Store data or get prediction from the model
        print(data)


    def on_error(self, status_code):
        print('Encountered error with status code:', status_code)
        return True # Don't kill the stream

    def on_timeout(self):
        print('Timeout...')
        return True # Don't kill the stream


# %%
if __name__ == "__main__":
    
    import settings

    # Keys for twitter API authentification 
    consumer_key = settings.consumer_key
    consumer_secret = settings.consumer_secret
    access_key = settings.access_key
    access_secret = settings.access_secret

    # Authentification process
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # Create SentimentStreamListener 
    sapi = tweepy.streaming.Stream(auth, SentimentStreamListener())

    # Apply filter of what should be listening
    sapi.filter(track=['bitcoin'], languages=['en'])