'''
Timeseries utilities
'''

import datetime
import enum
import logging
import typing


import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import d3m.container as container
import d3m.metadata.base as mbase

_logger = logging.getLogger()


class TimeIndicatorType(enum.Enum):
    OTHER = 0
    INTEGER = 1
    YEAR = 2
    YEAR_MONTH = 3
    MONTH_DAY = 4
    YEAR_MONTH_DAY = 5
    MONTH_DAY_YEAR = 6

# TimeIndicatorType.YEAR
# dateTime
# 0,1818,9.2,213.0,52.9
# 1,1819,7.9,249.0,38.5
# 2,1820,6.4,224.0,24.2
# 3,1821,4.2,304.0,9.2
# 4,1822,3.7,353.0,6.3
# 5,1823,2.7,302.0,2.2

# from 56_sunspots
# TimeIndicatorType.YEAR_MONTH
# dateTime
# 0,1749-01,96.7
# 1,1749-02,104.3
# 2,1749-03,116.7
# 3,1749-04,92.8
# 4,1749-05,141.7
# 5,1749-06,139.2

# TimeIndicatorType.INTEGER
# integer
# 0,cas9_VBBA,S_3102,4,28810
# 1,cas9_VBBA,S_3102,7,28869
# 2,cas9_VBBA,S_3102,8,28241
# 3,cas9_VBBA,S_3102,9,28399
# 4,cas9_VBBA,S_3102,10,28480
# 5,cas9_VBBA,S_3102,11,28695

# LL1_terra_canopy_height
# TimeIndicatorType.INTEGER
# integer
# d3mIndex,cultivar,sitename,day,canopy_height
# 0,PI145619,MAC Field Scanner Season 4 Range 27 Column 11,12,10
# 1,PI145619,MAC Field Scanner Season 4 Range 27 Column 11,13,10
# 2,PI145619,MAC Field Scanner Season 4 Range 27 Column 11,14,10
# 3,PI145619,MAC Field Scanner Season 4 Range 27 Column 11,15,10
# 4,PI145619,MAC Field Scanner Season 4 Range 27 Column 11,16,10
# 5,PI145619,MAC Field Scanner Season 4 Range 27 Column 11,17,10


# TimeIndicatorType.MONTH_DAY
# d3mIndex,Company,Year,Date,Close
# 0,abbv,2013,01-04,28.81
# 1,abbv,2013,01-07,28.869
# 2,abbv,2013,01-08,28.241999999999997
# 3,abbv,2013,01-09,28.399
# 4,abbv,2013,01-10,28.480999999999998
# 5,abbv,2013,01-11,28.695

# New date format: LL1_736_stock_market
# d3mIndex,Company,Date,Close
# 0,abbv,1/4/2013,28.81
# 1,abbv,1/7/2013,28.869
# 2,abbv,1/8/2013,28.242
# 3,abbv,1/9/2013,28.399
# 4,abbv,1/10/2013,28.481
# 5,abbv,1/11/2013,28.695

# LL1_PHEM
# d3mIndex,RegionName,ZoneName,WoredaName,dateTime,Malnutrition_total Cases
# 0,Addis Ababa,Addis Ketema,Addis Ketema,2017-01-31,8
# 1,Addis Ababa,Addis Ketema,Addis Ketema,2017-02-28,5
# 2,Addis Ababa,Addis Ketema,Addis Ketema,2017-03-31,11
# 3,Addis Ababa,Addis Ketema,Addis Ketema,2017-04-30,16
# 4,Addis Ababa,Addis Ketema,Addis Ketema,2017-05-31,15
# 5,Addis Ababa,Addis Ketema,Addis Ketema,2017-06-30,19

TIME_TYPE = 'https://metadata.datadrivendiscovery.org/types/Time'
INTEGER_TYPE = 'http://schema.org/Integer'
FLOAT_TYPE = 'http://schema.org/Float'
CATEGORICAL_TYPE = 'https://metadata.datadrivendiscovery.org/types/CategoricalData'
TRUE_TARGET_TYPE = 'https://metadata.datadrivendiscovery.org/types/TrueTarget'
SUGGESTED_GROUPING_KEY = "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"


def next_month(value: datetime.datetime) -> datetime.datetime:
    '''Increment by one month'''
    year = value.year
    month = value.month
    if value.month == 12:
        year = year + 1
        month = 1
    else:
        month += 1

    # May fail if time_tuple.tm_mday > 28
    return value.replace(year=year, month=month)


def next_day(value: datetime.datetime) -> datetime.datetime:
    '''Increment by one day'''
    day = 1 + value.day
    try:
        value = value.replace(day=day)
    except ValueError:
        day = 1
        value = value.replace(day=day)
        value = next_month(value)
    return value


def next_week(value: datetime.datetime) -> datetime.datetime:
    '''Increment by one week'''
    for i in range(7):
        value = next_day(value)
    return value


class TimeIndicator:
    '''
    Given d3m dataframe, detect is time indicator column style (TimeIndicatorType).
    '''
    def __init__(self, training_data: container.DataFrame):
        time_indicator_index = training_data.metadata.get_columns_with_semantic_type(TIME_TYPE)
        if len(time_indicator_index) > 1:
            _logger.warn(f'More than one time indicator columns. Using first one: {time_indicator_index}')
        self.time_indicator_index: int = time_indicator_index[0]
        self.indicator_style, self.separator = self._deduce_style(training_data, self.time_indicator_index)

        self.year_index = -1
        if self.indicator_style == TimeIndicatorType.MONTH_DAY:
            self.year_index = self._find_year_index(training_data)

        if self.indicator_style in [TimeIndicatorType.YEAR_MONTH_DAY, TimeIndicatorType.MONTH_DAY_YEAR]:
            first = self.get_datetime(training_data.iloc[0, :])
            second = self.get_datetime(training_data.iloc[1, :])
            step = second - first
            self.by_week = step.days == 7
            _logger.debug(f'By week {step}: {self.by_week}')


    def get_datetime(self, row: pd.Series) -> typing.Union[int, datetime.date]:
        '''
        Returns datetime associated with a row.
        '''
        row_list = row.tolist()
        value = row_list[self.time_indicator_index]
        if self.indicator_style == TimeIndicatorType.INTEGER:
            return int(value)
        if self.indicator_style == TimeIndicatorType.YEAR:
            return datetime.date(int(value), 1, 1)
        fields = [int(x) for x in value.split(self.separator)]
        if self.indicator_style == TimeIndicatorType.YEAR_MONTH:
            return datetime.date(fields[0], fields[1], 1)
        if self.indicator_style == TimeIndicatorType.MONTH_DAY:
            if self.year_index == -1:
                # No year, just pick a leap year
                year = 2000
            else:
                year = int(row_list[self.year_index])
            return datetime.date(year, fields[0], fields[1])
        if self.indicator_style == TimeIndicatorType.YEAR_MONTH_DAY:
            return datetime.date(fields[0], fields[1], fields[2])
        if self.indicator_style == TimeIndicatorType.MONTH_DAY_YEAR:
            return datetime.date(fields[2], fields[0], fields[1])
        return None

    def get_datetimes(self, data: container.DataFrame) -> typing.List[typing.Union[int, datetime.date]]:
        '''
        Returns all datetimes associated with the dataframe
        '''
        return [self.get_datetime(data.iloc[idx, :]) for idx in range(data.shape[0])]

    def get_next_time(self, date: typing.Union[int, datetime.date]) -> typing.Union[int, datetime.date]:
        '''
        Returns next time period based on time indicator style.
        '''
        if self.indicator_style == TimeIndicatorType.INTEGER:
            return date+1
        if self.indicator_style == TimeIndicatorType.YEAR:
            return date.replace(year=date.year+1)
        if self.indicator_style == TimeIndicatorType.YEAR_MONTH:
            return next_month(date)
        if self.indicator_style in [TimeIndicatorType.MONTH_DAY, TimeIndicatorType.YEAR_MONTH_DAY, TimeIndicatorType.MONTH_DAY_YEAR]:
            if self.by_week:
                return next_week(date)
            else:
                return next_day(date)

        return next_day(date)

    def get_difference(self, start, end) -> int:  # start, end: typing.Union[int, datetime.date]
        '''
        Difference between two dates based on time indicator style.
        '''
        if self.indicator_style == TimeIndicatorType.INTEGER:
            return end - start
        if self.indicator_style == TimeIndicatorType.YEAR:
            return end.year - start.year
        delta = end - start
        if self.indicator_style == TimeIndicatorType.YEAR_MONTH:
            return round(delta.days/30.4375)
        if self.indicator_style in [TimeIndicatorType.MONTH_DAY, TimeIndicatorType.YEAR_MONTH_DAY, TimeIndicatorType.MONTH_DAY_YEAR]:
            return delta.days
        return delta.days

    def get_date_range(self, first, last) -> pd.Index:
        '''
        Range between two dates based on time indicator style.
        '''
        # first = self.get_datetime(data.iloc[0, :])
        # second = self.get_datetime(data.iloc[1, :])
        # last = self.get_datetime(data.iloc[-1, :])
        if self.indicator_style == TimeIndicatorType.INTEGER:
            return pd.Index(list(range(first, last+1)))

        periods = 1 + self.get_difference(first, last)
        if self.indicator_style == TimeIndicatorType.YEAR:
            return pd.date_range(first, periods=periods, freq='YS')
        if self.indicator_style == TimeIndicatorType.YEAR_MONTH:
            return pd.date_range(first, periods=periods, freq='MS')

        if self.by_week:
            return pd.date_range(first, periods=periods, freq='W')
        else:
            return pd.date_range(first, periods=periods, freq='D')

    def _deduce_style(self, training_data: container.DataFrame, time_indicator_index: int) -> typing.Tuple[TimeIndicatorType, str]:
        metadata = training_data.metadata.query([mbase.ALL_ELEMENTS, time_indicator_index])
        if INTEGER_TYPE in metadata['semantic_types']:
            return TimeIndicatorType.INTEGER, ''

        column = training_data.iloc[:, time_indicator_index]
        if len(column) > 50:
            subset_index = np.random.choice(len(column), 50, replace=False)
        else:
            subset_index = range(len(column))
        values = column.iloc[subset_index]
        values = [value for value in values if isinstance(value, str)]

        # Detect separator
        possible_chars = ['-', '/']
        counts = []
        for char in possible_chars:
            counts.append(sum([value.count(char) for value in values]))
        separator = possible_chars[counts.index(max(counts))]

        # Need to check for str, missing value is np.nan (float)
        fields = [value.split(separator) for value in values if isinstance(value, str)]

        field_counts = np.array([len(field) for field in fields])
        if np.all(field_counts == 1):
            # No sepataor in any fields, i.e. one field
            return TimeIndicatorType.YEAR, separator

        if np.all(field_counts == 2):

            field0 = np.array([int(x[0]) for x in fields])

            if np.any(field0 > 12):
                return TimeIndicatorType.YEAR_MONTH, separator
            else:
                return TimeIndicatorType.MONTH_DAY, separator

        if np.all(field_counts == 3):
            if separator == '/':
                return TimeIndicatorType.MONTH_DAY_YEAR, separator
            else:
                return TimeIndicatorType.YEAR_MONTH_DAY, separator

        return TimeIndicatorType.OTHER, separator


    def _find_year_index(self, training_data: container.DataFrame) -> int:
        categorical_indices = training_data.metadata.get_columns_with_semantic_type(CATEGORICAL_TYPE)
        for index in categorical_indices:
            metadata = training_data.metadata.query([mbase.ALL_ELEMENTS, index])
            if metadata['structural_type'] is str:
                if self._string_column_is_year(training_data.iloc[:, index]):
                    return index
            elif metadata['structural_type'] is int:
                if self._int_column_is_year(training_data.iloc[:, index]):
                    return index
        return -1

    def _string_column_is_year(self, column) -> bool:
        if len(column) > 50:
            subset_index = np.random.choice(len(column), 50)
        else:
            subset_index = range(len(column))
        try:
            for index in subset_index:
                value = column[index]
                if int(value) > 9999:
                    return False
        except Exception:
            return False
        return True

    def _int_column_is_year(self, column) -> bool:
        if len(column) > 50:
            subset_index = np.random.choice(len(column), 50)
        else:
            subset_index = range(len(column))
        for index in subset_index:
            value = column[index]
            if value > 9999 or value < 1:
                return False
        return True


class ExtractTimeseries():
    '''
    Given data with muliple embedded timeseries, extract individual timeseries values. In addition the missing values
    are filled using 'pad'.
    '''
    def __init__(self, data, time_indicator):
        self.data = data
        self.time_indicator = time_indicator

        # Assume only on target
        self.target_index = data.metadata.get_columns_with_semantic_type(TRUE_TARGET_TYPE)[0]
        self.feature_indices = data.metadata.list_columns_with_semantic_types([FLOAT_TYPE, INTEGER_TYPE])

        if self.target_index in self.feature_indices:
            self.feature_indices.remove(self.target_index)

        self.categorical_indices = data.metadata.get_columns_with_semantic_type(SUGGESTED_GROUPING_KEY)
        # if time_indicator.year_index > -1:
        #     self.categorical_indices.remove(time_indicator.year_index)
        #     self.categorical_indices.append(time_indicator.year_index)
        self.categorical_column_name = [self.data.columns[i] for i in self.categorical_indices]

    def groupby(self) -> typing.Tuple[typing.Tuple, pd.DataFrame]:
        '''
        '''
        if self.categorical_column_name:
            for name, group in self.data.groupby(self.categorical_column_name):
                yield name, self.expand_fill(group)
        else:
            yield 'only_one_time_series', self.expand_fill(self.data)

    def expand_fill(self, data: pd.DataFrame) -> pd.DataFrame:
        # add datetime index
        data.index = self.time_indicator.get_datetimes(data)

        # Remove duplicate dates
        duplicated = data.index.duplicated()
        if duplicated.sum() > 0:
            _logger.warning('Timeseries has duplicate dates: ' + str(data.head()))
            data = data[data.index.duplicated() == False]

        # Make sure dates are in order
        data = data.loc[data.index.sort_values()]

        targets = data.iloc[:, self.feature_indices + [self.target_index]]

        # redindex to add missing rows
        index2 = self.time_indicator.get_date_range(
            self.time_indicator.get_datetime(data.iloc[0, :]),
            self.time_indicator.get_datetime(data.iloc[-1, :]))

        targets = targets.reindex(index2, method='pad')
        return targets

    def combine(self, prediction_groups: typing.Dict, inputs: container.DataFrame):
        all_results = []

        only_one = 'only_one_time_series' in prediction_groups
        for i, row in inputs.iterrows():
            # date = pd.Timestamp(et.time_indicator.get_datetime(row))
            date = self.time_indicator.get_datetime(row)
            if only_one:
                key = 'only_one_time_series'
            else:
                key = []
                for x in self.categorical_indices:
                    key.append(row.iloc[x])
                key = tuple(key)
            predictions = prediction_groups[key]
            all_results.append(predictions.loc[date, 0])
        return np.array(all_results).T


        # all_results = []
        # for i, row in inputs.iterrows():
        #     date = ti.get_datetime(row)
        #     key = tuple(row.iloc[x] for x in [1,2])
        #     predictions = results[key]
        #     all_results.append(predictions[date])


class MultiVariableTimeseries():
    '''
    Class to convert spawn dataset-like dataset, i.e. multiple one-variable timeseries with each timeseries described
    by categorical attributes, to a multi-variable timeseries with categorical attribtues as columns and time as index.

    From:

    >>> data.head()
       d3mIndex    species  sector  day  count
    0         0  cas9_VBBA  S_3102    4  28810
    1         1  cas9_VBBA  S_3102    7  28869
    2         2  cas9_VBBA  S_3102    8  28241
    3         3  cas9_VBBA  S_3102    9  28399
    4         4  cas9_VBBA  S_3102   10  28480

    To:

    >>> time_vectors.head()
         (cas9_CAD, S_0102)  (cas9_CAD, S_1102)  (cas9_CAD, S_2102)         ...           (cas9_YABE, S_8002)  (cas9_YABE, S_9002)  (cas9_YABE, S_9991)
    day                                                                     ...
    2                   NaN                 NaN                 NaN         ...                       13674.0               6170.0                  NaN
    3                   NaN              3910.0              3410.0         ...                       13822.0                  NaN                  NaN
    4                4580.0              4019.0              3800.0         ...                       13173.0                  NaN               4208.0
    5                4730.0              3940.0              3810.0         ...                           NaN               6216.0               3931.0
    6                4860.0              3900.0              3630.0         ...                           NaN               6388.0               4962.0

    To access a column: time_vectors[('cas9_CAD', 'S_0102')]

    Example usage:
    data = pd.read_csv(os.path.join(dataset_base,'seed_datasets_current/LL1_736_population_spawn/LL1_736_population_spawn_dataset/tables/learningData.csv'))
    categorical_indicies = [1, 2]
    time_index = 3
    value_indices = 4
    mv = MultiVariableTimeseries(data, categorical_indicies, time_index, value_indices)
    mv.get_dataframe()

    '''
    def __init__(self, data, categorical_indicies: typing.List[int], time_index: int, value_index: int):
        self.data = data
        self.categorical_indicies = categorical_indicies
        self.time_index = time_index
        self.value_index = value_index
        self.range = self._get_time_range(data.iloc[:, time_index])

        # Maps column tuple name to integer, e.g. ('cas9_CAD', 'S_0102') maps to 0.
        self.column_index: typing.Dict = {}

    @staticmethod
    def _get_time_range(time_values: pd.Series) -> range:
        '''
        Given a series of time values returns range covering those time values
        '''
        time_sorted = time_values.unique()
        time_sorted.sort()
        min_time = time_sorted[0]
        max_time = time_sorted[-1]
        deltas = [b-a for a, b in zip(time_sorted, time_sorted[1:])]
        counts = pd.value_counts(deltas)
        min_delta = min(deltas)
        most_frequent_delta = counts.index[0]
        if min_delta != most_frequent_delta:
            _logger.warn(f'Min time delta != most frequent time delta: {min_delta} != {most_frequent_delta}')
        time_range = range(min_time, max_time+1, min_delta)
        # print(time_range.start, time_range.stop, time_range.step)
        return time_range

    def get_dataframe(self) -> pd.DataFrame:
        categorical_column_name = [self.data.columns[i] for i in self.categorical_indicies]
        column_names = []
        groups = {}
        result = []
        for name, group in self.data.groupby(categorical_column_name):
            column_names.append(name)
            result.append(group)
            groups[name] = group
        time_range = self._get_time_range(self.data.iloc[:, self.time_index])

        complete_index = pd.Index(time_range, name=self.data.columns[self.time_index])
        time_vectors = pd.DataFrame(index=complete_index, columns=column_names)
        for i, (name, group) in enumerate(self.data.groupby(categorical_column_name)):
            self.column_index[name] = i
            col = pd.DataFrame(group.iloc[:, self.value_index])
            col = col.set_index(group.iloc[:, self.time_index])
            col.reindex(complete_index)
            time_vectors.iloc[:, i] = col.iloc[:, 0]
        return time_vectors
