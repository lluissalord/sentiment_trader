import pickle
import os
from collections import Counter

from utils import fillAllTime

import numpy as np
import pandas as pd
import pandas_ta as ta

## Description of several TA features and their classification

# Price differences
# Awesome Oscillator (AO) --> Difference of prices
# Absolute Price Oscillator (APO) --> Difference of prices
# TRUERANGE --> Difference of prices
# Average True Range (ATR) --> Difference of prices
# Detrend Price Oscillator (DPO) --> Difference of prices
# Moving Average, Convergence/Divergence (contains MACD) --> Difference of prices (MACDH is difference of difference of prices)
# Momentum (MOM) --> Difference of prices
# Q Stick (QS) --> Difference of prices


# Volume
# AD --> ((close - open) * volume / (high - low + eps)).cumsum() --> somehow cumsum of weighted volume --> grows constantly
# 'OBV' in col  --> cumsum of signed volume
# Elder's Force Index (EFI) --> (Difference of prices) * Volumen --> Unbound
# Ease of Movement (EOM) --> (Unbound) Ponderation of how is moving high and low. Needs correct 'divisor' adapted to the volume
# Negative Volume Index (NVI) --> (Unbound) CumSum de volumen negativo por ROC
# Positive Volume Index (PVI) --> (Unbound) CumSum de volumen positivo por ROC
# Price-Volume (PVOL) --> (Unbound) Prices * Volumen
# Price-Volume Trend (PVT) --> (Unbound) CumSum of ROC * Volume

# Others
# CCI --> (Difference of prices) / c * MAD(mean price) --> where c is coefficient (default 0.015) and MAD is similar to std
# CG --> Value between 0 and length (default 10)
# Fisher Transform (FISHT) --> (Unbound) Because depends on the std dev of the market price, but it is important to see signals
# Mass Index (MASSI) --> (Unbound) However proportional and centered to slow (default 25) --> Substract and Divide by slow could be a solution
# Log Return (LOGRET) --> Logaritmic of (current price / previous price)
# Slope (SLOPE) --> Difference of prices / length --> Option would be to use as_angle=True which put it as radians
# Vortex (contains VTX) --> abs(Difference of prices) / abs(Difference of prices) --> (Unbound) > 0 centered around a bit less than 1

# Bounded index
# Rate of Change (ROC) --> -100 - 100
# Coppock Curve (COPC) --> -200 - 200 (weighted double of ROC)
# 'Know Sure Thing' (contains KST) --> -100000 - 100000 (weighted 1000 time of ROC) 
# Normalized Average True Range (NATR) --> 0 - 100
# Percent Return (PCTRET) --> -1 - 1
# Percentage Price Oscillator (PPO) --> -100 - 100
# Trix (TRIX)--> -100 - 100
# True Strength Index (TSI) --> -100 - 100
# Ultimate Oscillator (UO) --> -100 - 100
# William's Percent R (WILLR) --> -100 - 0

# Price ranges
# Kaufman's Adaptive Moving Average (KAMA) --> Price range

# Stadistical measures
# Kurtosis (KURT) --> (Unbound) It is a stadistical measure like skewness to measure tail (so if experiment extrem returns +/-)
# Skew (SKEW) --> (Unbound) It is a stadistical measure to measure tail (so if experiment extrem returns +/-)
# STDEV --> It is a stadistical measure --> Difference of prices
# Mean Absolute Deviation (MAD) --> It is a stadistical measure --> Difference of prices
# Variance (VAR) --> It is a stadistical measure --> Difference of prices
# Z Score (Z) --> Price normalized by Z score

eps = 1e-4

KNOWN_COLS = {
    '0_1': {
        'cols': [],
        'add_cols': True,
        'normalize': True,
        'max': 1,
        'min': 0,
        'std': eps,
    },
    '-1_1': {
        'cols': ['PCTRET'],
        'add_cols': True,
        'normalize': True,
        'max': 1,
        'min': -1,
        'std': eps,
    },
    '0_100': {
        'cols': ['NATR', ],
        'add_cols': True,
        'normalize': True,
        'max': 100,
        'min': 0,
        'std': eps,
    },
    '-100_0': {
        'cols': ['WILLR'],
        'add_cols': True,
        'normalize': True,
        'max': 0,
        'min': -100,
        'std': eps,
    },
    '-100_100': {
        'cols': ['ROC', 'PPO', 'PPOH', 'PPOS', 'TRIX', 'TSI', 'UO'],
        'add_cols': True,
        'normalize': True,
        'max': 100,
        'min': -100,
        'std': eps,
    },
    '-200_200': {
        'cols': ['COPC'],
        'add_cols': False,
        'normalize': True,
        'max': 200,
        'min': -200,
        'std': eps,
    },
    '-100000_100000': {
        'cols': ['KST'],
        'add_cols': False,
        'normalize': True,
        'max': 100000,
        'min': -100000,
        'std': eps,
    },
    'diff_prices': {
        'cols': ['AO', 'APO', 'ATR', 'DPO', 'MACD', 'MACDH', 'MACDS', 'MOM', 'QS'],
        'add_cols': False,
        'normalize': False,
    },
    'prices': {
        'cols': ['KAMA'],
        'add_cols': True,
        'ref_col': 'close',
        'normalize': False,
    },
}


def classifyColsByRanges(data, known_cols_dict=KNOWN_COLS):

    def removeFromAllColumns(all_columns, columns):
        for col in columns:
            try:
                i = all_columns.index(col)
                del all_columns[i]
            except ValueError:
                print(f'Column {col} not found in {all_columns}')

    # Transform all abreviated columns into the ones in data   
    ranges_dict = known_cols_dict.copy()
    all_known_columns = []
    for key, values in ranges_dict.items():
        clean_cols = values['cols']
        range_cols = []
        for clean_col in clean_cols:
            columns = list(data.columns[data.columns.str.startswith(clean_col+'_')])
            all_known_columns += columns
            range_cols += columns
        ranges_dict[key]['cols'] = range_cols

    all_columns = list(data.columns)

    # Remove all the already known columns
    removeFromAllColumns(all_columns, all_known_columns)

    for key in ranges_dict:
        columns = []
        if ranges_dict[key]['add_cols']:
            # Get parameters of current range
            if key == 'prices':
                max_ = data[ranges_dict[key]['ref_col']].max()
                min_ = data[ranges_dict[key]['ref_col']].min()
                std_ = data[ranges_dict[key]['ref_col']].std()
                ranges_dict[key]['max'] = max_
                ranges_dict[key]['min'] = min_
                ranges_dict[key]['std'] = std_
            else:
                max_ = ranges_dict[key]['max']
                min_ = ranges_dict[key]['min']
                std_ = ranges_dict[key]['std']

            # Extract columns which match the specification
            columns = list(data[all_columns].dtypes[(data.max() <= max_ + std_) & (data.min() >= min_ - std_)].index)

            # Remove from all_columns
            removeFromAllColumns(all_columns, columns)
            
            # Add sorted unique columns to the pertinent range
            ranges_dict[key]['cols'] = sorted(list(set(columns + ranges_dict[key]['cols'])))

    ranges_dict['others'] = {}
    ranges_dict['others']['cols'] = sorted(all_columns)
    ranges_dict['others']['normalize'] = False 

    return ranges_dict


def normalizeFeatures(data, ranges_dict):
    
    for _, values in ranges_dict.items():
        if values['normalize']:
            columns = values['cols']
            max_ = values['max']
            min_ = values['min']
            data[columns] = (data[columns] - min_) / (max_ - min_)

    return data


def generateTAFeatures(data, freq='D', exclude_ind=[], args=None):
    # Indicators not posible to use 'short_run' and 'cross'
    not_ind = ['long_run', 'short_run', 'cross']
    not_ind.append('ichimoku') # Output has to be treaten different because is returning two DataFrames
    not_ind.append('trend_return') # Required trend column

    # Add exclude_ind
    not_ind += exclude_ind

    indicators = [ind for ind in data.ta.indicators(as_list=True) if ind not in not_ind]

    if args is None:
        basic_args = {'append': True, 'ewm': True, 'adjust': True, 'freq': freq}
        basic_args = dict(zip(indicators, [basic_args] * len(indicators)))

        args = basic_args

    # TODO: Implement short-term and long-term arguments for each indicator

    n_args = len(args)
    for i, (ind, arg) in enumerate(args.items()):
        print(f'{i} out of {n_args} features', end='\r')
        data.ta(kind=ind, **arg)

    return data


def cleanNan(data):
    # Drop columns which have all columns as NaN
    remove_cols = data.dtypes[data.isnull().all()].index
    if len(remove_cols) > 0:
        data = data.drop(remove_cols, axis=1)
        print(f'The following columns have been removed: {list(remove_cols)}')

    # Drop rows which have at least one NaN
    print(f'Dropping {data.isnull().any(axis=1).sum()} rows because of NaN values')
    data = data[data.notnull().all(axis=1)]

    return data


def main(prices_path, ranges_dict_path, save_path, onlyRead=True, cleanNans=True, exclude_ind=[], args=None, freq='min', freq_raw='min', sep=',', timestamp_col=None, timestamp_unit='s', start_date=None, end_date=None, columns_dict=None):
    """Preprocess on prices historical data filling up all entries, aggregating by frequency, treating NA and differenciating
    """

    if onlyRead and os.path.exists(ranges_dict_path) and os.path.exists(save_path):
        ranges_dict_path = 'data\\ranges_dict.pickle'

        data = pd.read_csv(save_path, sep='\t', index_col=timestamp_col)
        data = data.set_index(
            pd.to_datetime(data.index)
        )
        with open(ranges_dict_path, 'rb') as f:
            ranges_dict = pickle.load(f)

    else:

        print('Loading data...')
        parse_dates = timestamp_col is not None

        raw_df = pd.read_csv(
            prices_path,
            sep=sep,
            index_col=timestamp_col,
            parse_dates=parse_dates
        )

        # Transform Timestamp, which is expressed in seconds, to index
        raw_df = raw_df.set_index(
            pd.to_datetime(raw_df.index, unit=timestamp_unit)
        )

        # Filter to tret only between start_date and end_date
        if start_date is None:
            start_date = raw_df.index.min()
        if end_date is None:
            end_date = raw_df.index.max()
        raw_df = raw_df[
            (raw_df.index >= start_date)
            & (raw_df.index <= end_date)
        ]

        # Generate columns_dict to rename columns in case it does not exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if columns_dict is None:
            columns_dict = {}
            lower_cols = [col.lower() for col in list(raw_df.columns)]
            for col in required_columns:
                for i, lower_col in enumerate(lower_cols):
                    if col in lower_col:
                        orig_col = list(raw_df.columns)[i]
                        columns_dict[orig_col] = col
                        break

        # Dictionary columns_dict must have the values of the required_columns
        assert Counter(required_columns) == Counter(list(columns_dict.values())), f'Dictionary columns_dict must have the following values: {required_columns}'

        # Rename the corresponding columns to 'open', 'high', 'low', 'close' and 'volume'
        raw_df = raw_df.rename(columns=columns_dict)

        print("Filling All Time data")
        # Fill all the seconds between first and last second of data
        df = fillAllTime(
            raw_df,
            freq=freq_raw
        )

        print("Filling NA data")
        # As null data is due to no transaction on that minute (or in minimal cases shutdown of API)
        # Means that the prices is the same as in the previous minute
        df = df.fillna(method='ffill')

        if freq_raw != freq:
            print(f"Aggregating from {freq_raw} to {freq} level")
            # Aggregate by frequency taking into account columns: open, high, low, close, volume
            data = df.resample(freq).agg(
                {
                    'open': lambda x: x.iloc[0],
                    'high': 'max',
                    'low': 'min',
                    'close': lambda x: x.iloc[-1],
                    'volume': 'sum'
                }
            )
        else:
            data = df

        # Only required columns are used
        data = data[required_columns]

        print('Generating TA features...')
        data = generateTAFeatures(data, 1, exclude_ind, args) # TA Features directly with default values, not adapted
        data = generateTAFeatures(data, freq, exclude_ind, args) # TA Features adapted to the current frequency
        ranges_dict = classifyColsByRanges(data)
        data = normalizeFeatures(data, ranges_dict)
        data = cleanNan(data)
        data.to_csv(save_path, sep='\t', index_label='Timestamp')

        with open(ranges_dict_path, 'wb') as f:
            pickle.dump(ranges_dict, f)

    return data, ranges_dict