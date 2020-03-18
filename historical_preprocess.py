# %%
import pandas as pd
import numpy as np
import time
import os

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

def fillAllTime(df, freq='min', on=None, keep='first', start_dt=None, end_dt=None):
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


def tweetsPreprocess(tweets_path, freq='min', nrows=None, chunksize=5e5, save_path='data/preprocess/twitter.csv', write_files=True):
    
    print("Loading raw file")
    raw_df_chunks = pd.read_csv(
        tweets_path,
        sep=';',
        usecols=['timestamp', 'replies', 'likes', 'retweets', 'text'],
        nrows=nrows,
        chunksize=chunksize
    )

    agg_list = []
    saved_files = []
    for i, raw_df in enumerate(raw_df_chunks):
        print(f"Processing chunk {i}...")
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
        raw_df['timestamp'] = raw_df['timestamp'].dt.floor(freq)

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

        if write_files:
            partial_file = os.path.splitext(save_path)
            saved_file = f'{partial_file[0]}_{i}{partial_file[1]}'
            agg_df.to_csv(saved_file, sep='\t')
            saved_files.append(saved_file)
        else:
            # Add to list of data
            agg_list.append(agg_df)

    print("All chunks processed")
    if write_files:
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

        agg_df = pd.read_csv(save_path, sep='\t', index_col='timestamp')
        agg_df = agg_df.set_index(
            pd.to_datetime(agg_df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        )

    else:
        agg_df = pd.concat(agg_list)

    print("Filling All Time data")
    # Fill all the seconds between first and last second of data
    df = fillAllTime(
        agg_df,
        freq=freq
    )

    df['no_tweets'] = df.iloc[:,0].isnull().astype('int8')
    df = df.fillna(0)

    return df

def pricesPreprocess(prices_path, freq='min'): 

    print("Loading raw file")
    raw_df = pd.read_csv(
        prices_path,
        sep=',',
        usecols=['Timestamp','Close', 'Volume_(BTC)'],
        index_col='Timestamp',
        parse_dates=True
    )
    raw_df = raw_df.set_index(
        pd.to_datetime(raw_df.index, unit='s')
    )

    print("Filling All Time data")
    # Fill all the seconds between first and last second of data
    df = fillAllTime(
        raw_df,
        freq=freq
    )

    print("Filling NA data")
    # As null data is due to no transaction on that minute (or in minimal cases shutdown of API)
    # Means that the prices is the same as in the previous minute
    df = df.fillna(method='ffill')

    # Use difference with previous price instead of absolute value
    df['Close'] = df['Close'].diff()
    df = df.iloc[1:]

    return df

# %%
%%time
tweets_path = 'data/tweets_historical.csv'
prices_path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv'
tweets_df = tweetsPreprocess(tweets_path, nrows=1000, freq='H')
prices_df = pricesPreprocess(prices_path, freq='H')
all_df = prices_df.merge(tweets_df, how='left', left_index=True, right_index=True)
all_df['no_data'] = all_df[tweets_df.columns[0]].isnull()
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
    all_df['no_data'] = all_df[tweets_df.columns[0]].isnull()
    all_df.to_csv('data/all_data.csv', sep='\t')
