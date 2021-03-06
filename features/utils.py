# -*- coding: utf-8 -*-
""" Utils for feature engineering """

import numpy as np
import pandas as pd

from scipy.stats import normaltest
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from unidip import UniDip

import re
import string

from nltk import download

from textblob import TextBlob
from nltk.corpus import stopwords
download('stopwords')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
download('vader_lexicon')

#For TextBlob ------------------------
# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':&gt;', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '&gt;:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '&gt;:)', '&gt;;)', '&gt;:-)',
    '&lt;3'
    ])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '&gt;:/', ':S', '&gt;:[', ':@', ':-(', ':[', ':-||', '=L', ':&lt;',
    ':-[', ':-&lt;', '=\\', '=/', '&gt;:(', ':(', '&gt;.&lt;', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '&gt;:\\', ';('
    ])

#Emoji patterns
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols &amp; pictographs
                           u"\U0001F680-\U0001F6FF"  # transport &amp; map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)


def blob_clean_tweets(tweet):
    """ Remove undesired tokens and clean tweet text using Blob package and regex """

    stop_words = set(stopwords.words('english'))

    #after tweepy preprocessing the colon left remain after removing mentions
    #or RT sign in the beginning of the tweet
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)

    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)

    #remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)

    blob = TextBlob(tweet)
    word_tokens = blob.words

    #filter using NLTK library append it to a string
    #filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []

    #looping through conditions
    for w in word_tokens:
        #check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)


def blobSentimentAnalyser(text):
    """ Extract sentiment output (polarity and subjectivity) from raw tweet using Blob package """

    blob = TextBlob(blob_clean_tweets(text))
    #pass textBlob method for sentiment calculations
    Sentiment = blob.sentiment

    #seperate polarity and subjectivity in to two variables
    return Sentiment.polarity, Sentiment.subjectivity

# ------------------------

# For VADER  ------------------------
def _remove_pattern(input_txt, pattern, replace=''):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, replace, input_txt)        
    return input_txt
    
def vader_clean_tweets(tweet):
    """ Remove undesired tokens from tweets, useb by VADER package """

    # remove twitter Return handles (RT @xxx:)
    tweet = _remove_pattern(tweet, r"RT @[\w]*:")
    # remove twitter handles (@xxx)
    tweet = _remove_pattern(tweet, r"@[\w]*")
    # remove URL links (httpxxx)
    tweet = _remove_pattern(tweet, r"https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    tweet = re.sub(r"[^a-zA-Z#]", " ", tweet)

    return tweet

def vaderSentimentAnalyser(text, analyser):
    """ Extract sentiment scores (neg, neu, pos, compound) from raw tweet, using VADER package"""

    new_text = vader_clean_tweets(text)

    scores = analyser.polarity_scores(new_text)
    return scores

def vec_vader_clean_tweets(lst):
    """ Vectorized function for removing undesired tokens from tweets, useb by VADER package """

    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(_remove_pattern)(lst, r"RT @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(_remove_pattern)(lst, r"@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(_remove_pattern)(lst, r"https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, r"[^a-zA-Z#]", " ")
    
    return lst


def vec_vaderSentimentAnalyser(text_lst, analyser, rename_dict=None):
    """ Vectorized extraction of sentiment scores (neg, neu, pos, compound) from raw tweet, using VADER package"""

    new_text_lst = vec_vader_clean_tweets(text_lst)

    def polarity_scores(text):
        scores = analyser.polarity_scores(text)
        
        if rename_dict is not None:
            scores[rename_dict['neg']] = scores.pop('neg')
            scores[rename_dict['neu']] = scores.pop('neu')
            scores[rename_dict['pos']] = scores.pop('pos')
            scores[rename_dict['compound']] = scores.pop('compound')
        return scores

    scores = np.vectorize(polarity_scores)(new_text_lst)

    return scores

# ------------------------

def fillAllTime(df, freq='min', on=None, keep='first', start_dt=None, end_dt=None):
    """Creates DataFrame with all the time steps in df[on] or between start_dt and end_dt """
    df_copy = df.copy()
    if on is None:
        df_copy.index = df_copy.index.floor(freq)
        df_copy = df_copy[~df_copy.index.duplicated(keep=keep)]
        if start_dt is None:
            start_dt = df_copy.index.values[0]
        if end_dt is None:
            end_dt = df_copy.index.values[-1]

    else:
        df_copy[on] = df_copy[on].dt.floor(freq)
        df_copy = df_copy.drop_duplicates(on, keep=keep)
        if start_dt is None:
            start_dt = df_copy[on].values[0]
        if end_dt is None:
            end_dt = df_copy[on].values[-1]

    # Fill all the seconds between first and last second of data
    all_mins = pd.DataFrame(
        index=pd.date_range(
            start=start_dt,
            end=end_dt,
            freq=freq
        )
    )

    df_copy.index = df_copy.index.tz_convert(None)

    return all_mins.merge(
        df_copy,
        how='left',
        left_index=True,
        right_index=on is None,
        right_on=on,
    )


def normalize_data(df, p_val_threshold=0.05, quantile_threshold=1000, max_quantiles=200, min_points_per_quantile=10):
    """ Normalize to zero-mean and unit variance the continuos data using different transformation depending on the data distribution """
    
    fitted_df = df.copy()

    # Define number of quantiles for the transformation
    n_quantiles = min(len(df.index)//min_points_per_quantile, max_quantiles)

    transformers = {}

    # Calculate statistic normality tests for all float columns
    float_cols = df.select_dtypes(include=[np.float]).columns
    stats, p_values = normaltest(df[float_cols])
    for i, column in enumerate(float_cols):
        p_val = p_values[i]
        # Determine if data follows a Gaussian distribution
        if p_val < p_val_threshold:
            # For large datasets is best to use directly Quantile even if meaning is lost
            if len(df.index) > quantile_threshold:
                # Use Quantile Transform for multimodal
                transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')
            else:
                try:
                    # Test multimodality getting intervals of peaks
                    intervals = UniDip(df[column], debug=False).run()

                    # Only 1 peak means it is unimodal, otherwise multimodal 
                    if len(intervals) == 1:
                        transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                    else:
                        # Use Quantile Transform for multimodal
                        transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')
                except IndexError:
                    transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            # When data is following a Gaussian it is used a simple zero-mean unit variance transformation
            transformer = StandardScaler(with_mean=True, with_std=True)

        fitted_df[column] = transformer.fit_transform((df[column]).values.reshape(-1,1))
        transformers[column] = transformer
    
    return fitted_df, transformers