import numpy as np

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


def blob_clean_tweets(tweet, clean_analysis=True):

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
    blob = TextBlob(blob_clean_tweets(text))
    #pass textBlob method for sentiment calculations
    Sentiment = blob.sentiment

    #seperate polarity and subjectivity in to two variables
    return Sentiment.polarity, Sentiment.subjectivity

# ------------------------

# For VADER  ------------------------
def remove_pattern(input_txt, pattern, replace=''):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, replace, input_txt)        
    return input_txt
    
def vader_clean_tweets(tweet):
    # remove twitter Return handles (RT @xxx:)
    tweet = remove_pattern(tweet, r"RT @[\w]*:")
    # remove twitter handles (@xxx)
    tweet = remove_pattern(tweet, r"@[\w]*")
    # remove URL links (httpxxx)
    tweet = remove_pattern(tweet, r"https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    tweet = re.sub(r"[^a-zA-Z#]", " ", tweet)

    return tweet

def vaderSentimentAnalyser(text, analyser):
    new_text = vader_clean_tweets(text)

    scores = analyser.polarity_scores(new_text)
    return scores

def vec_vader_clean_tweets(lst):
    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(remove_pattern)(lst, r"RT @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(remove_pattern)(lst, r"@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(remove_pattern)(lst, r"https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, r"[^a-zA-Z#]", " ")
    
    return lst


def vec_vaderSentimentAnalyser(text_lst, analyser, rename_dict=None):
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