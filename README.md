# Sentiment trader

Trader which take into account twitter sentiment on specific topic and Technical Analysis to decide to go Long or Short.

## Package Overview

Currently this package has three main pilars: (1) Feature engineering on tweet sentiments and on prices using Technical Analysis; (2) Reinforcement learning environment for trading and (3) Jupyter Notebook which join all this parts and train an RL DNN model.

## Installation

In order to install the required packages, please create an Conda environment from `environment.yml` using the following command:

```
conda env create -n <ENV NAME> -f environment.yml
```

Besides, to configure the Tweepy streaming, you will need to add some tokens on `settings.py.default`. There are plenty of guides on Internet on how to get these Tweeter API tokens ([here](https://elfsight.com/blog/2020/03/how-to-get-twitter-api-key/) you can find an example). Then add them on `settings.py.default` and once done, please rename the file by `settings.py`.