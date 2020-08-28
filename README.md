# Sentiment trader

Trader which take into account twitter sentiment on specific topic and Technical Analysis to decide to go Long or Short.

## Package Overview

Currently this package has three main pilars: (1) Feature engineering on tweet sentiments and on prices using Technical Analysis; (2) Reinforcement learning environment for trading and (3) Jupyter Notebook which join all this parts and train an RL DNN model.

API Documentation [here](https://lluissalord.github.io/sentiment_trader/index.html)

## Installation

In order to install the required packages, please create an Conda environment from `environment.yml` using the following command:

```
conda env create -n <ENV NAME> -f environment.yml
```

Besides, to configure the Tweepy streaming, you will need to add some tokens on `settings.py.default`. There are plenty of guides on Internet on how to get these Tweeter API tokens ([here](https://elfsight.com/blog/2020/03/how-to-get-twitter-api-key/) you can find an example). Then add them on `settings.py.default` and once done, please rename the file by `settings.py`.

## Data sources

In order to run Jupyter Notebook presented here, some CSV files are required. These have been extracted from Kaggle and they have to be allocated on `data/source/` folder. The CSV files used are:

* Tweets about Bitcoin: https://www.kaggle.com/alaix14/bitcoin-tweets-20160101-to-20190329
* Bitcoin prices: https://www.kaggle.com/mczielinski/bitcoin-historical-data#bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv
* Other historical prices: https://www.dukascopy.com/trading-tools/widgets/quotes/historical_data_feed