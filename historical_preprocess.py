# %%
import pandas as pd
import numpy as np
import time
import os

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

def fillAllTime(df, freq='min', on=None, keep='first', start_dt=None, end_dt=None):
    """Creates DataFrame with all the time steps in df[on] or between start_dt and end_dt
    """
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
    """Returns input DataFrame adding VADER sentiment columns
    """
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
    """Returns input DataFrame adding TextBlob sentiment columns
    """
    from utils import blobSentimentAnalyser

    sentiment_list = np.vectorize(blobSentimentAnalyser)(df['text'])
    new_df = pd.DataFrame(sentiment_list, index=TEXTBLOB_COLUMNS, columns=df.index).T

    return pd.concat([df, new_df], axis=1)


def weight_mean(x, df, weight_col, offset=0):
    """General function to calculate weighted mean depending on 'weight_col'
    """
    weights = df.loc[x.index, weight_col] + offset
    if sum(weights) == 0:
        return 0
    return np.average(x, weights=weights)


def tweetsPreprocess(tweets_path, freq='min', use_vader=True, use_textBlob=True, start_date=None, end_date=None, nrows=None, chunksize=5e5, save_path='data/preprocess/twitter.csv', write_files=True):
    """Preprocess on tweet historical data which adds sentiment columns and aggregate them depending on different weight columns by frequency
    """
    AGG_COLUMNS = ['replies', 'likes', 'retweets']

    print("Loading raw file")
    raw_df_chunks = pd.read_csv(
        tweets_path,
        sep=';',
        usecols=['timestamp', 'text'] + AGG_COLUMNS,
        nrows=nrows,
        chunksize=chunksize,
        engine='python',
    )

    all_list = []
    saved_files = []
    for i, raw_df in enumerate(raw_df_chunks):
        print(f"Processing chunk {i}...")
        print("Transforming timestamp to datetime format")
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], format='%Y-%m-%d %H:%M:%S+00')

        print("Filtering rows")
        # Filter to tret only valid data between start_date and end_date
        raw_df = raw_df[
            (raw_df['timestamp'].notnull())
            & (raw_df['text'].notnull())
            & ((start_date is None) | (raw_df['timestamp'] >= start_date))
            & ((end_date is None) | (raw_df['timestamp'] <= end_date))
        ]

        print(f"After filter there are {len(raw_df.index)} rows out of {chunksize:.0f}")
        if len(raw_df.index) == 0:
            continue

        SENTIMENT_COLUMNS = []
        
        if use_vader:
            print("Adding VADER Sentiment")
            raw_df = addVaderSentiment(raw_df)
            SENTIMENT_COLUMNS = list(set(SENTIMENT_COLUMNS + VADER_COLUMNS))

        if use_textBlob:
            print("Adding TextBlob Sentiment")
            raw_df = addTextBlobSentiment(raw_df)
            SENTIMENT_COLUMNS = list(set(SENTIMENT_COLUMNS + TEXTBLOB_COLUMNS))

        columns = ['timestamp'] + AGG_COLUMNS + SENTIMENT_COLUMNS

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
        all_df['timestamp'] = pd.to_datetime(all_df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    else:
        all_df = pd.concat(all_list)

    # Floor timestamp at freq level to make sure that aggregation process is done correctly
    print("Flooring timestamp")
    all_df['timestamp'] = all_df['timestamp'].dt.floor(freq)

    # Define simple operation to do on AGG_COLUMNS when aggregating by freq
    func_dict = dict(
        zip(
            AGG_COLUMNS,
            [['sum','mean'],] * 3
        )
    )

    if use_vader or use_textBlob:
        # Define weighted means to do on SENTIMENT_COLUMNS when aggregating by freq
        replies_mean = lambda x: weight_mean(x, all_df, weight_col='replies', offset=0)
        likes_mean = lambda x: weight_mean(x, all_df, weight_col='likes', offset=0)
        retweets_mean = lambda x: weight_mean(x, all_df, weight_col='retweets', offset=0)
        func_dict.update(
            dict(
                zip(
                    SENTIMENT_COLUMNS,
                    [['mean', replies_mean, likes_mean, retweets_mean],] * len(SENTIMENT_COLUMNS)
                )
            )
        )

    print("Aggregating by timestamp")
    agg_df = all_df.groupby(['timestamp'])[columns].agg(func_dict)
    
    # As weighted means are defined by lambda functions, these have to be renamed
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
        freq=freq
    )

    # Add 0/1 column defining each freq when it has not been tweets
    df['no_tweets'] = df.iloc[:,0].isnull().astype('int8')

    # Filling nulls (frequencies where it has not been tweets) with 0's
    df = df.fillna(0)

    return df

def pricesPreprocess(prices_path, freq='min', start_date=None, end_date=None, rolling_window=60*24*7): 
    """Preprocess on prices historical data filling up all entries, aggregating by frequency, treating NA and differenciating
    """
    print("Loading raw file")
    raw_df = pd.read_csv(
        prices_path,
        sep=',',
        usecols=['Timestamp','Close', 'Volume_(BTC)'],
        index_col='Timestamp',
        parse_dates=True
    )

    # Transform Timestamp, which is expressed in seconds, to index
    raw_df = raw_df.set_index(
        pd.to_datetime(raw_df.index, unit='s')
    )

    # Filter to tret only between start_date and end_date
    raw_df = raw_df[
        ((start_date is None) | (raw_df.index >= start_date))
        & ((end_date is None) | (raw_df.index <= end_date))
    ]

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
    df['Close_diff'] = df['Close'].diff()
    df = df.iloc[1:]

    # Calculate moving average on rolling window
    df['Close_moving_average'] = df['Close_diff'].rolling(rolling_window).mean()

    # Rename Volume_(BTC) to Volume_BTC
    df.columns = ['Close', 'Close_diff', 'Volume_BTC', 'Close_moving_average']

    return df

# %%
%%time
if __name__ == "__main__":
    tweets_path = 'data/sources/tweets_historical.csv'
    prices_path = 'data/sources/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv'

    start_date='2019-01-01'
    end_date='2019-01-05'

    freq = 'min'

    print("Start tweetsPreprocess")
    tweets_df = tweetsPreprocess(
        tweets_path,
        freq=freq,
        use_vader=False,
        use_textBlob=False,
        start_date=start_date,
        end_date=end_date,
        nrows=None,
        chunksize=5e5,
        save_path='data/preprocess/twitter.csv',
        write_files=False
    )

    print("Start pricesPreprocess")
    prices_df = pricesPreprocess(
        prices_path,
        freq=freq,
        start_date=start_date,
        end_date=end_date,
        rolling_window=60*24,
    )

    print("Joining prices and tweets")
    all_df = prices_df.merge(tweets_df, how='left', left_index=True, right_index=True)

    # TODO: Review if still makes sense or it is always 0
    # all_df['no_data'] = all_df[tweets_df.columns[0]].isnull()
    # all_df[tweets_df.columns] = all_df[tweets_df.columns].fillna(0).astype('int8')

    # Store final data
    all_df.to_csv('data/all_data.csv', sep='\t', index_label='Timestamp')

# %%
