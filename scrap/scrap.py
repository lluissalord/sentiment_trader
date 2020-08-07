import GetOldTweets3 as got
import datetime
import time
from tqdm import tqdm

def scrapAllTweetsByTime(query, start_date, end_date=None, datetime_format='%Y-%m-%d', freq='days', freq_value=1):
    """ Scrap tweets fullfiling query between specified dates and downloading them in packages with the specified frequency """

    # Transform string datetime to datetime object
    st_datetime = datetime.datetime.strptime(start_date, datetime_format)
    if end_date is not None:
        end_datetime = datetime.datetime.strptime(end_date, datetime_format)
    else:
        today = datetime.datetime.today()
        end_datetime = datetime.datetime(today.year, today.month, today.day)

    # Set frequency as timedelta and adapt end datetime to the frequency
    if freq.lower() in ['d', 'day', 'days']:
        delta = datetime.timedelta(days=freq_value)
    elif freq.lower() in ['h', 'hour', 'hours']:
        delta = datetime.timedelta(hours=freq_value)
        end_datetime += datetime.timedelta(hours=1) * today.hour
    elif freq.lower() in ['m', 'minute', 'minutes', 'min', 'mins']:
        delta = datetime.timedelta(minutes=freq_value)
        end_datetime += datetime.timedelta(minutes=1) * today.minute
    else:
        raise ValueError(f'Frequency should be "days", "hours" or "minutes", but it is "{freq}"')
    

    # Creates the list of datetimes defining the packages to download
    dates = []
    current_datetime = min(st_datetime + delta, end_datetime)
    while current_datetime != end_datetime:
        dates.append(current_datetime)

        current_datetime += delta
        current_datetime = min(current_datetime + delta, end_datetime)
    
    # Download each package depending on the datetime
    all_tweets = []
    for current_datetime in tqdm(dates):
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query)\
                                                    .setSince(current_datetime.isoformat())\
                                                    .setUntil((current_datetime + delta).isoformat())\
                                                    # .setMaxTweets(max_tweets)\
                                                    # .setMinFaves(5)\ # Only available on repo version
                                                    # .setTopTweets(True)\
                                           
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        all_tweets += tweets

    return all_tweets