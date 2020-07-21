""" Price feature engineering """

import pickle
import os
from collections import Counter

from features.utils import fillAllTime

import numpy as np
import pandas as pd
import pandas_ta as ta

eps = 1e-4

# There are some features which range of values is already known and estimation is not required
# Besides, this dictionary helps to know how to normalize each feature
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
        'cols': ['AO', 'APO', 'ATR', 'DPO', 'MACDH', 'MACDS', 'MOM', 'QS'], # 'MACD'
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

# Transform from a frequency name (standard for pandas) to a description
FREQUENCY_DESCRIPTION = {
    's': 'sec',
    'm': 'min',
    'h': 'hour',
    'd': 'day',
}


def classifyColsByRanges(data, known_cols_dict=KNOWN_COLS):
    """ Classify each column from data into a group from KNOWN_COLS """

    def _removeFromAllColumns(all_columns, columns):
        for col in columns:
            try:
                i = all_columns.index(col)
                del all_columns[i]
            except ValueError:
                print(f'Column {col} not found in {all_columns}')

    # Transform all abreviated columns in `know_cols` into the full name column names in data   
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
    _removeFromAllColumns(all_columns, all_known_columns)

    # Look for columns which match requirements for each group
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
            _removeFromAllColumns(all_columns, columns)
            
            # Add sorted unique columns to the pertinent range
            ranges_dict[key]['cols'] = sorted(list(set(columns + ranges_dict[key]['cols'])))

    # Final group with the rest of columns which have not been classified in a group
    ranges_dict['others'] = {}
    ranges_dict['others']['cols'] = sorted(all_columns)
    ranges_dict['others']['normalize'] = False 

    return ranges_dict


def normalizeFeatures(data, ranges_dict):
    """ Apply min-max normalization on all the features which have normalization range """
    
    for _, values in ranges_dict.items():
        if values['normalize']:
            columns = values['cols']
            max_ = values['max']
            min_ = values['min']
            data[columns] = (data[columns] - min_) / (max_ - min_)

    return data


def generateTAFeatures(data, exclude_ind=[], args=None, suffix=''):
    """ Generate all the Technical Analysis features excluding the specified ones """

    # Indicators not posible to use 'short_run' and 'cross'
    not_ind = ['long_run', 'short_run', 'cross']
    not_ind.append('ichimoku') # Output has to be treaten different because is returning two DataFrames
    not_ind.append('trend_return') # Required trend column

    # Add exclude_ind
    not_ind += exclude_ind

    indicators = [ind for ind in data.ta.indicators(as_list=True) if ind not in not_ind]

    if args is None:
        basic_args = {'append': True, 'ewm': True, 'adjust': True, 'suffix': suffix}
        basic_args = dict(zip(indicators, [basic_args] * len(indicators)))

        args = basic_args

    # TODO: Implement short-term and long-term arguments for each indicator

    n_args = len(args)
    for i, (ind, arg) in enumerate(args.items()):
        print(f'{i} out of {n_args} features', end='\r')
        data.ta(kind=ind, **arg)

    return data


def cleanNan(data):
    """ Clean data from NaN, columns with all NaNs and rows with at least one NaN """

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
    """ Preprocess on prices historical data filling up all entries, aggregating by frequency, treating NA and differenciating """

    if onlyRead and os.path.exists(ranges_dict_path) and os.path.exists(save_path):
        ranges_dict_path = 'data\\ranges_dict.pickle'

        final_data = pd.read_csv(save_path, sep='\t', index_col=timestamp_col)
        final_data = final_data.set_index(
            pd.to_datetime(final_data.index)
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

        if type(freq) is str:
            freq = [freq]
        
        all_data = []
        for frequency in freq:
            if freq_raw != frequency:
                print(f"Aggregating from {freq_raw} to {frequency} level")
                # Aggregate by frequency taking into account columns: open, high, low, close, volume
                data = df.resample(frequency).agg(
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
            all_data.append((data, frequency))

        print('Generating TA features...')
        for i, (data, frequency) in enumerate(all_data):

            frequency_descr = FREQUENCY_DESCRIPTION[frequency]
            if args is not None:
                for ind in args:
                    args[ind]['suffix'] = frequency_descr
            
            ta_data = generateTAFeatures(data, exclude_ind, args, suffix=frequency_descr)

            if i == 0:
                final_data = ta_data
            else:
                final_data = pd.merge_asof(final_data, ta_data, left_index=True, right_index=True, suffixes=('','_'))

                # Remove original columns (open, low, high, close, volume) from the right side
                final_data = final_data[final_data.columns[~final_data.columns.str.endswith('_')]]

        ranges_dict = classifyColsByRanges(final_data)
        final_data = normalizeFeatures(final_data, ranges_dict)
        final_data = cleanNan(final_data)
        final_data.to_csv(save_path, sep='\t', index_label='Timestamp')

        with open(ranges_dict_path, 'wb') as f:
            pickle.dump(ranges_dict, f)

    return final_data, ranges_dict