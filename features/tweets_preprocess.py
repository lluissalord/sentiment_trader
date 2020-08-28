""" Tweet sentiment analysis feature engineering """

import pandas as pd
import numpy as np
import time
import os
import glob
from collections import Counter

from features.utils import fillAllTime

# Default datetime format
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S+00'

# Sentiment column names extracted from VADER process
VADER_COLUMNS = [
    'Negative',
    'Neutral',
    'Positive',
    'Compound',
]

# Sentiment column names extracted from TextBlob process
TEXTBLOB_COLUMNS = [
    'Polarity',
    'Subjectivity',
]


def addVaderSentiment(df, cols=VADER_COLUMNS):
    """Returns input DataFrame adding VADER sentiment columns
    """
    from features.utils import vec_vaderSentimentAnalyser
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    if len(df.index) == 0:
        return df

    analyser = SentimentIntensityAnalyzer()
    
    scores = vec_vaderSentimentAnalyser(
        df['text'],
        analyser,
        rename_dict=dict(
            zip(
                ['neg','neu','pos','compound'],
                VADER_COLUMNS
            )
        )
    )

    new_df = pd.DataFrame.from_records(
        scores,
        index=df.index
    )[cols]

    return pd.concat([df, new_df], axis=1)

def addTextBlobSentiment(df, cols=TEXTBLOB_COLUMNS):
    """Returns input DataFrame adding TextBlob sentiment columns
    """
    from features.utils import blobSentimentAnalyser

    if len(df.index) == 0:
        return df
    
    sentiment_list = np.vectorize(blobSentimentAnalyser)(df['text'])
    new_df = pd.DataFrame(
        sentiment_list,
        index=TEXTBLOB_COLUMNS,
        columns=df.index
    ).T[cols]

    return pd.concat([df, new_df], axis=1)


def weight_mean(x, df, weight_col, offset=0):
    """General function to calculate weighted mean depending on 'weight_col'
    """
    weights = df.loc[x.index, weight_col] + offset
    if sum(weights) == 0:
        return 0
    return np.average(x, weights=weights)


def date_filter(df, start_date=None, end_date=None, timestamp_col='timestamp', datetime_format=DATETIME_FORMAT):
    """ Filter DataFrame between dates """

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=datetime_format)

    print("Filtering rows")
    # Filter to tret only valid data between start_date and end_date
    min_time = df[timestamp_col].min()
    max_time = df[timestamp_col].max()
    print(f'Min timestamp {min_time}, Max timestamp {max_time}')

    df = df[
        (df[timestamp_col].notnull())
        & (df['text'].notnull())
        & ((start_date is None) | (df[timestamp_col] >= start_date))
        & ((end_date is None) | (df[timestamp_col] <= end_date))
    ]
    
    return df


def calculate_sentiment(df, sentiment_cols):
    """ Calculate sentiment of each tweet on DataFrame """

    # Check and store columns used in sentiment_cols
    vader_cols = [col for col in sentiment_cols if col in VADER_COLUMNS]
    use_vader = len(vader_cols) > 0

    textBlob_cols = [col for col in sentiment_cols if col in TEXTBLOB_COLUMNS]
    use_textBlob = len(textBlob_cols) > 0

    # TODO: Add moving average on sentiment
    # Process VADER sentiment only if required
    if use_vader:
        print("Adding VADER Sentiment")
        df = addVaderSentiment(df, vader_cols)
        sentiment_cols = list(set(sentiment_cols + vader_cols))

    # Process TextBlob sentiment only if required
    if use_textBlob:
        print("Adding TextBlob Sentiment")
        df = addTextBlobSentiment(df, textBlob_cols)
        sentiment_cols = list(set(sentiment_cols + textBlob_cols))

    return df, sentiment_cols, use_textBlob, use_vader


def tweet_featuring(df, aggregate_cols, sentiment_cols, use_textBlob, use_vader):
    """ Perform featuring on tweets numerical data and aggregate it at current time level """

    # TODO: Add parametrizable 'aggregate_func' instead of directly ['sum','mean']
    # Define simple operation to do on 'aggregate_cols' when aggregating by freq
    func_dict = dict(
        zip(
            aggregate_cols,
            [['sum','mean'],] * len(aggregate_cols)
        )
    )

    if use_vader or use_textBlob:
        # Define weighted means to do on 'sentiment_cols' when aggregating by freq
        weight_cols = aggregate_cols
        # TODO: Add parametrizable 'agg_sentiment_func' instead of directly ['mean']
        agg_sentiment_func = ['mean']
        replace_dict = {}
        for i, weight_col in enumerate(weight_cols):
            agg_sentiment_func.append(
                lambda x: weight_mean(x, df, weight_col=weight_col, offset=0)
            )

            replace_dict[f'<lambda_{i}>'] = f'{weight_col}_mean'

        func_dict.update(
            dict(
                zip(
                    sentiment_cols,
                    [agg_sentiment_func,] * len(sentiment_cols)
                )
            )
        )

    print("Aggregating by timestamp")
    agg_df = df.groupby(['timestamp']).agg(func_dict)

    # As weighted means are defined by lambda functions, these have to be renamed
    agg_df.columns = list(
        agg_df.columns.to_frame()
            .replace(replace_dict)
            .agg('_'.join, axis=1)
    )

    return agg_df


def preprocess_tweet_level(df, sentiment_cols, start_date, end_date, timestamp_col, datetime_format=DATETIME_FORMAT):
    """ Preprocess raw tweets data which can be treated at tweet level """

    # Filter DataFrame to only include data between start_date and end_date
    df = date_filter(df, start_date=start_date, end_date=end_date, timestamp_col=timestamp_col, datetime_format=datetime_format)

    # Run sentiment on the dataset
    df, sentiment_cols, use_textBlob, use_vader = calculate_sentiment(df, sentiment_cols)

    return df, sentiment_cols, use_textBlob, use_vader


def final_preprocess(df, freq, aggregate_cols, sentiment_cols, use_textBlob, use_vader, start_date, end_date, timestamp_col, save_path, save_final_df=True, sep='\t', save_path_add_date=True):
    """ Preprocess data containing already sentiment values, generate new features and make sure of data consistency """    

    # Floor timestamp at freq level to make sure that aggregation process is done correctly
    print("Flooring timestamp")
    df[timestamp_col] = df[timestamp_col].dt.floor(freq)

    # Process tweets to aggregate DataFrame at timestamp (freq) level
    agg_df = tweet_featuring(df, aggregate_cols, sentiment_cols, use_textBlob, use_vader)

    print("Filling All Time data")
    # Fill all the seconds between first and last second of data
    df = fillAllTime(
        agg_df,
        freq=freq
    )

    # Add 0/1 column defining each freq when it has not been tweets
    df['no_tweets'] = df.iloc[:,0].isnull().astype('int8')

    # Filling nulls (frequencies where it has not been tweets) with 0's
    df = df.fillna(0)

    if save_final_df:
        partial_file = os.path.splitext(save_path)
        if save_path_add_date:
            save_final_path = f'{partial_file[0]}_{start_date}_-_{end_date}{partial_file[1]}'
        else:
            save_final_path = f'{partial_file[0]}{partial_file[1]}'
        df.to_csv(save_final_path, sep=sep, index_label=timestamp_col)

    return df


def tweets_preprocess(df, freq='min', sentiment_cols=VADER_COLUMNS+TEXTBLOB_COLUMNS, aggregate_cols=['replies', 'likes', 'retweets'], start_date=None, end_date=None, timestamp_col='timestamp', datetime_format=DATETIME_FORMAT, save_path='data/preprocess/twitter.csv', save_final_df=True, save_path_add_date=True):
    """Preprocess tweets adding sentiment columns, creating features and aggregating them at frequency level """

    initial_lenght = len(df.index)
    df, sentiment_cols, use_textBlob, use_vader = preprocess_tweet_level(df, sentiment_cols, start_date, end_date, timestamp_col, datetime_format)

    print(f"After filter there are {len(df.index)} rows out of {initial_lenght}")
    if len(df.index) == 0:
        print(f"No data between starting date '{start_date}' and ending date '{end_date}'")
        return df

    df = final_preprocess(df, freq, aggregate_cols, sentiment_cols, use_textBlob, use_vader, start_date, end_date, timestamp_col, save_path, save_final_df, sep='\t', save_path_add_date=save_path_add_date)

    return df


def chunk_tweets_preprocess(tweets_path, freq='min', sentiment_cols=VADER_COLUMNS+TEXTBLOB_COLUMNS, aggregate_cols=[], start_date=None, end_date=None, timestamp_col='timestamp', datetime_format=DATETIME_FORMAT, nrows=None, chunksize=5e5, save_path='data/preprocess/twitter.csv', write_files=True, save_final_df=True, save_path_add_date=True):
    """Preprocess on tweet historical data in chunks, adding sentiment columns and aggregating them depending on different weight columns by frequency
    """

    print("Loading raw file")
    raw_df_chunks = pd.read_csv(
        tweets_path,
        sep=';',
        usecols=[timestamp_col, 'text'] + aggregate_cols,
        nrows=nrows,
        chunksize=chunksize,
        engine='python',
    )

    all_list = []
    saved_files = []
    for i, raw_df in enumerate(raw_df_chunks):
        print(f"Processing chunk {i}...")

        raw_df, sentiment_cols, use_textBlob, use_vader = preprocess_tweet_level(raw_df, sentiment_cols, start_date, end_date, timestamp_col, datetime_format)

        print(f"After filter there are {len(raw_df.index)} rows out of {chunksize:.0f}")
        if len(raw_df.index) == 0:
            continue

        # Summarise all columns which are stored
        columns = [timestamp_col] + aggregate_cols + sentiment_cols

        # Use write_files when data is too large to fit in-memory
        if write_files:
            partial_file = os.path.splitext(save_path)
            saved_file = f'{partial_file[0]}_{i}{partial_file[1]}'
            raw_df[columns].to_csv(saved_file, sep='\t')
            saved_files.append(saved_file)
        else:
            # Add to list of data
            all_list.append(raw_df[columns])

    print("All chunks processed")

    print("Concatenating all the chunks")
    if write_files:
        # Join all CSV files by manually adding all in one file
        if os.path.exists(save_path):
            os.remove(save_path)
        fout=open(save_path,"a")
        # first file:
        for line in open(saved_files[0]):
            fout.write(line)
        # now the rest:    
        if len(saved_files) > 1:
            for save_file in saved_files[1:]:
                f = open(save_file)
                for n, line in enumerate(f):
                    if n != 0 and line not in ['', '\n']:
                        fout.write(line)
                f.close() # not really needed
        fout.close()

        # Read file with all the chunks together
        all_df = pd.read_csv(save_path, sep='\t', usecols=columns)
        all_df[timestamp_col] = pd.to_datetime(all_df[timestamp_col], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # Remove temporary files
        for f in saved_files:
            os.remove(f)
        os.remove(save_path)

    else:
        all_df = pd.concat(all_list)

    df = final_preprocess(all_df, freq, aggregate_cols, sentiment_cols, use_textBlob, use_vader, start_date, end_date, timestamp_col, save_path, save_final_df, sep='\t', save_path_add_date=save_path_add_date)

    return df

# TODO: Could be interesting to get number of followers of the username
def extract_tweets(tweets):
    """ Transform data from list of Tweet objectes to DataFrame """
    # Currently only using timestamp and text
    # Data from favorites, replies and retweets differ between historical and inference 
    data = {
        'timestamp': [],
        'text': [],
        # 'favorites': [],
        # 'replies': [],
        # 'retweets': [],
        # 'mentions': [],
        # 'hashtags': [],
        # 'urls': [],
    }
    for tweet in tweets:
        data['timestamp'].append(tweet.date)
        data['text'].append(tweet.text)
        # data['favorites'].append(tweet.favorites)
        # data['replies'].append(tweet.replies)
        # data['retweets'].append(tweet.retweets)
        # data['mentions'].append(len(tweet.mentions.split(' ')))
        # data['hashtags'].append(len(tweet.hashtags.split(' ')))
        # data['urls'].append(len(tweet.urls.split(',')))

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Scrapped from twitters from 2016-01-01 to 2019-03-29, Collecting Tweets containing Bitcoin or BTC
    tweets_path = '../data/sources/tweets_historical.csv'

    start_date='2019-01-01'
    end_date='2019-01-02'

    freq = 'h'

    # TODO: Save tweets sentiment independent of prices and one file per date range and frequency

    print("Start chunk_tweets_preprocess")
    tweets_df = chunk_tweets_preprocess(
        tweets_path,
        freq=freq,
        # sentiment_cols=VADER_COLUMNS+TEXTBLOB_COLUMNS,
        # sentiment_cols=['Compound', 'Polarity'],
        aggregate_cols=['replies', 'likes', 'retweets'],
        start_date=start_date,
        end_date=end_date,
        nrows=80000,
        chunksize=5e4,
        save_path='../data/preprocess/twitter_test.csv',
        write_files=False
    )
