# import numpy as np

# # TODO: Use https://github.com/twopirllc/pandas-ta as Technical Analysis library?

# # TODO: Support/Resistance trending line (should be aX + b, hence 2 features?)

# # TODO: Marks of (stochastic?) crossover between trends and prices (with 0/1 or wit value between [0-1])
# # Formula on https://www.investopedia.com/terms/s/stochasticoscillator.asp

# # TODO: StochRSI is interesting to have
# # Link https://www.investopedia.com/terms/s/stochrsi.asp

# # TODO: Add MAC D

# # TODO: If we can could number of exchanges, then we can calculate TRIN (Trading index)

# from historical_preprocess import pricesFeatureExtraction

# def difference(dataDf, period):
#     return dataDf.sub(dataDf.shift(period))
    
# def ewm(dataDf, halflife):
#     return dataDf.ewm(halflife=halflife, ignore_na=False,
#                       min_periods=0, adjust=True).mean()

# def rsi(data, period):
#     data_upside = data.sub(data.shift(1), fill_value=0)
#     data_downside = data_upside.copy()
#     data_downside[data_upside > 0] = 0
#     data_upside[data_upside < 0] = 0
#     avg_upside = data_upside.rolling(period).mean()
#     avg_downside = - data_downside.rolling(period).mean()
#     rsi = 100 - (100 * avg_downside / (avg_downside + avg_upside))
#     rsi[avg_downside == 0] = 100
#     rsi[(avg_downside == 0) & (avg_upside == 0)] = 0
    
#     return rsi

# def bollinger_band(df, col, period, no_of_std=2):

#     #Calculate rolling mean and standard deviation using number of days set above
#     rolling_mean = df[col].rolling(period).mean()
#     rolling_std = df[col].rolling(period).std()

#     #create two new DataFrame columns to hold values of upper and lower Bollinger bands
#     df['Rolling Mean'] = rolling_mean
#     df['Bollinger High'] = rolling_mean + (rolling_std * no_of_std)
#     df['Bollinger Low'] = rolling_mean - (rolling_std * no_of_std)

#     return df

# # TODO: MACD
# # exp1 = df.y.ewm(span=12, adjust=False).mean()
# # exp2 = df.y.ewm(span=26, adjust=False).mean()
# # macd = exp1-exp2
# # exp3 = macd.ewm(span=9, adjust=False).mean()

# def normalize(df, col, rolling_window):
#     std = df[col].rolling(rolling_window).std()
#     std[std == 0] = 1
#     return (df[col] - df[col].rolling(rolling_window).mean()) / std

# def pricesFeatureExtraction(df, rolling_window, price_col='Close', eps=1e-4):
    
#     df[price_col+'_diff'] = difference(df[price_col], rolling_window)
#     df[price_col+'_diff'] = normalize(df, price_col+'_diff', rolling_window)

#     df[price_col+'_moving_average'] = ewm(df[price_col], rolling_window)
#     df[price_col+'_moving_average'] = normalize(df, price_col+'_moving_average', rolling_window)

#     df[price_col+'_rsi'] = rsi(df[price_col], rolling_window)
#     df[price_col+'_rsi'] = (df[price_col+'_rsi'] - 50) / 50   # Zero-centered and between [-1, 1]

#     df['rsi_buy'] = (df[price_col+'_rsi'] >= 30).astype(int)
#     df['rsi_sell'] = (df[price_col+'_rsi'] <= 70).astype(int)

#     df = bollinger_band(df, price_col, rolling_window)

#     return df

# # Z-score filter (explosive/peak points)
# class real_time_peak_detection():
#     """
#     Rules of thumb for configuring the algorithm

#     lag: the lag parameter determines how much your data will be smoothed and how adaptive the algorithm is to changes in the long-term average of the data. The more stationary your data is, the more lags you should include (this should improve the robustness of the algorithm). If your data contains time-varying trends, you should consider how quickly you want the algorithm to adapt to these trends. I.e., if you put lag at 10, it takes 10 'periods' before the algorithm's treshold is adjusted to any systematic changes in the long-term average. So choose the lag parameter based on the trending behavior of your data and how adaptive you want the algorithm to be.

#     influence: this parameter determines the influence of signals on the algorithm's detection threshold. If put at 0, signals have no influence on the threshold, such that future signals are detected based on a threshold that is calculated with a mean and standard deviation that is not influenced by past signals. Another way to think about this is that if you put the influence at 0, you implicitly assume stationarity (i.e. no matter how many signals there are, the time series always returns to the same average over the long term). If this is not the case, you should put the influence parameter somewhere between 0 and 1, depending on the extent to which signals can systematically influence the time-varying trend of the data. E.g., if signals lead to a structural break of the long-term average of the time series, the influence parameter should be put high (close to 1) so the threshold can adjust to these changes quickly.

#     threshold: the threshold parameter is the number of standard deviations from the moving mean above which the algorithm will classify a new datapoint as being a signal. For example, if a new datapoint is 4.0 standard deviations above the moving mean and the threshold parameter is set as 3.5, the algorithm will identify the datapoint as a signal. This parameter should be set based on how many signals you expect. For example, if your data is normally distributed, a threshold (or: z-score) of 3.5 corresponds to a signaling probability of 0.00047 (from this table), which implies that you expect a signal once every 2128 datapoints (1/0.00047). The threshold therefore directly influences how sensitive the algorithm is and thereby also how often the algorithm signals. Examine your own data and determine a sensible threshold that makes the algorithm signal when you want it to (some trial-and-error might be needed here to get to a good threshold for your purpose).
 
#     """
#     def __init__(self, array, lag, threshold, influence):
#         self.y = list(array)
#         self.length = len(self.y)
#         self.lag = lag
#         self.threshold = threshold
#         self.influence = influence
#         self.signals = [0] * len(self.y)
#         self.filteredY = np.array(self.y).tolist()
#         self.avgFilter = [0] * len(self.y)
#         self.stdFilter = [0] * len(self.y)
#         self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
#         self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()

#     def thresholding_algo(self, new_value):
#         self.y.append(new_value)
#         i = len(self.y) - 1
#         self.length = len(self.y)
#         if i < self.lag:
#             return 0
#         elif i == self.lag:
#             self.signals = [0] * len(self.y)
#             self.filteredY = np.array(self.y).tolist()
#             self.avgFilter = [0] * len(self.y)
#             self.stdFilter = [0] * len(self.y)
#             self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
#             self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()
#             return 0

#         self.signals += [0]
#         self.filteredY += [0]
#         self.avgFilter += [0]
#         self.stdFilter += [0]

#         if abs(self.y[i] - self.avgFilter[i - 1]) > self.threshold * self.stdFilter[i - 1]:
#             if self.y[i] > self.avgFilter[i - 1]:
#                 self.signals[i] = 1
#             else:
#                 self.signals[i] = -1

#             self.filteredY[i] = self.influence * self.y[i] + (1 - self.influence) * self.filteredY[i - 1]
#             self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
#             self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])
#         else:
#             self.signals[i] = 0
#             self.filteredY[i] = self.y[i]
#             self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
#             self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])

#         return self.signals[i]