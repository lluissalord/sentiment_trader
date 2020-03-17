# %%
import pandas as pd
import numpy as np
import time

VADER_COLUMNS = [
    'Negative',
    'Neutral',
    'Positive',
    'Compound',
]

TEXTBLOB_COLUMNS = [
    'Polarity',
    'Subjectivity',
]

def fillAllTime(df, freq='min', on=None, start_dt=None, end_dt=None):
    df_copy = df.copy()
    if on is None:
        df_copy.index = df_copy.index.floor(freq)
        if start_dt is None:
            start_dt = df.index.values[0]
        if end_dt is None:
            end_dt = df.index.values[-1]

    else:
        df_copy[on] = df_copy[on].dt.floor(freq)
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

    return all_mins.merge(
        df_copy,
        how='left',
        left_index=True,
        right_index=on is None,
        right_on=on,
    )


def addVaderSentiment(df):
    from utils import vec_vaderSentimentAnalyser
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
    )

    return pd.concat([df, new_df], axis=1)

def addTextBlobSentiment(df):
    from utils import blobSentimentAnalyser

    sentiment_list = np.vectorize(blobSentimentAnalyser)(df['text'])
    new_df = pd.DataFrame(sentiment_list, index=TEXTBLOB_COLUMNS, columns=df.index).T

    return pd.concat([df, new_df], axis=1)


def weight_mean(x, df, weight_col, offset=0):
    weights = df.loc[x.index, weight_col] + offset
    if sum(weights) == 0:
        return 0
    return np.average(x, weights=weights)


def tweetsPreprocess(tweets_path, nrows=None):
    
    print("Loading raw file")
    raw_df = pd.read_csv(tweets_path, sep=';', nrows=nrows)

    print("Transforming timestamp to datetime format")
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], format='%Y-%m-%d %H:%M:%S+00')

    raw_df = raw_df[
        (raw_df['timestamp'].notnull())
        & (raw_df['text'].notnull()) 
    ]

    print("Adding Sentiment")
    SENTIMENT_COLUMNS = []
    
    raw_df = addVaderSentiment(raw_df)
    SENTIMENT_COLUMNS += VADER_COLUMNS

    raw_df = addTextBlobSentiment(raw_df)
    SENTIMENT_COLUMNS += TEXTBLOB_COLUMNS

    func_dict = dict(
        zip(
            ['replies', 'likes', 'retweets'],
            [['sum','mean'],] * 3
        )
    )

    replies_mean = lambda x: weight_mean(x, raw_df, weight_col='replies', offset=0)
    likes_mean = lambda x: weight_mean(x, raw_df, weight_col='likes', offset=0)
    retweets_mean = lambda x: weight_mean(x, raw_df, weight_col='retweets', offset=0)

    func_dict.update(
        dict(
            zip(
                SENTIMENT_COLUMNS,
                [['mean', replies_mean, likes_mean, retweets_mean],] * len(SENTIMENT_COLUMNS)
            )
        )
    )

    print("Flooring timestamp")
    raw_df['timestamp'] = raw_df['timestamp'].dt.floor('min')

    columns = ['replies', 'likes', 'retweets'] 
    columns += SENTIMENT_COLUMNS

    print("Aggregating by timestamp")
    agg_df = raw_df.groupby(['timestamp'])[columns].agg(func_dict)
    
    agg_df.columns = list(
        agg_df.columns.to_frame()
            .replace('<lambda_0>', 'replies_mean')
            .replace('<lambda_1>', 'likes_mean')
            .replace('<lambda_2>', 'retweets_mean') 
            .agg('_'.join, axis=1)
    )

    print("Filling All Time data")
    # Fill all the seconds between first and last second of data
    df = fillAllTime(
        agg_df,
        freq='min'
    )

    return df

def pricesPreprocess(prices_path): 

    print("Loading raw file")
    raw_df = pd.read_csv(prices_path, sep=',', index_col='Timestamp', parse_dates=True)
    raw_df = raw_df.set_index(
        pd.to_datetime(raw_df.index, unit='s')
    )

    df = raw_df[['Close', 'Volume_(BTC)']]

    print("Filling All Time data")
    # Fill all the seconds between first and last second of data
    df = fillAllTime(
        df,
        freq='min'
    )

    print("Filling NA data")
    # As null data is due to no transaction on that minute (or in minimal cases shutdown of API)
    # Means that the prices is the same as in the previous minute
    df = df.fillna(method='ffill')

    return df

# %%
%%time
tweets_path = 'data/tweets_historical.csv'
prices_path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv'
tweets_df = tweetsPreprocess(tweets_path, nrows=10000)
prices_df = pricesPreprocess(prices_path)
all_df = prices_df.merge(tweets_df, how='left', left_index=True, right_index=True)
# %%
%%time
if __name__ == "__main__":
    tweets_path = 'data/tweets_historical.csv'
    prices_path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv'

    print("Start tweetsPreprocess")
    tweets_df = tweetsPreprocess(tweets_path)

    print("Start pricesPreprocess")
    prices_df = pricesPreprocess(prices_path)

    print("Joining prices and tweets")
    all_df = prices_df.merge(tweets_df, how='left', left_index=True, right_index=True)

# %%


# %%
