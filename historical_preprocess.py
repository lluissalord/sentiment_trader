# %%
import pandas as pd

from utils import vader_clean_tweets
from utils import blob_clean_tweets
from nltk.sentiment.vader import SentimentIntensityAnalyzer

path = 'data/tweets_historical.csv'
df = pd.read_csv(path, sep=';', nrows=100)

# %%
%%time
# VADER
analyser = SentimentIntensityAnalyzer()
def sentimentAnalyser(text, analyser):
    new_text = vader_clean_tweets(text)
    scores = analyser.polarity_scores(text)
    return scores['neg'], scores['neu'], scores['pos'], scores['compound']

neg, neu, pos, comp = [], [], [], []

for _, row in df.iterrows():
    neg_, neu_, pos_, comp_ = sentimentAnalyser(row.text, analyser)
    neg.append(neg_)
    neu.append(neu_)
    pos.append(pos_)
    comp.append(comp_)

new_df = pd.DataFrame(
    {
        'Negative': neg,
        'Neutral': neu,
        'Positive': pos,
        'Compound': comp
    },
    index=df.index
)

df = pd.concat([df, new_df], axis=1)


# %%
%%time
# TextBlob
def sentimentAnalyser(text):
    blob_text, blob = blob_clean_tweets(text)
    #pass textBlob method for sentiment calculations
    Sentiment = blob.sentiment

    #seperate polarity and subjectivity in to two variables
    return Sentiment.polarity, Sentiment.subjectivity

pol, sub = [], []

for _, row in df.iterrows():
    pol_, sub_ = sentimentAnalyser(row.text)
    pol.append(pol_)
    sub.append(sub_)

new_df = pd.DataFrame(
    {
        'Polarity': pol,
        'Subjectiviy': sub
    },
    index=df.index
)

df = pd.concat([df, new_df], axis=1)
# %%
df

# %%
