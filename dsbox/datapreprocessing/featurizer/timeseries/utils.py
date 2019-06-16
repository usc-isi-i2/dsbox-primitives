'''
Timeseries utilities
'''

import datetime
import enum
import logging
import typing


import numpy as np
import pandas as pd

import d3m.container as container
import d3m.metadata.base as mbase

_logger = logging.getLogger()


class TimeIndicatorType(enum.Enum):
    OTHER = 0
    INTEGER = 1
    YEAR = 2
    YEAR_MONTH = 3
    MONTH_DAY = 4

# TimeIndicatorType.YEAR
# dateTime
# 0,1818,9.2,213.0,52.9
# 1,1819,7.9,249.0,38.5
# 2,1820,6.4,224.0,24.2
# 3,1821,4.2,304.0,9.2
# 4,1822,3.7,353.0,6.3
# 5,1823,2.7,302.0,2.2

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

# TimeIndicatorType.MONTH_DAY
# d3mIndex,Company,Year,Date,Close
# 0,abbv,2013,01-04,28.81
# 1,abbv,2013,01-07,28.869
# 2,abbv,2013,01-08,28.241999999999997
# 3,abbv,2013,01-09,28.399
# 4,abbv,2013,01-10,28.480999999999998
# 5,abbv,2013,01-11,28.695


def next_month(value: datetime) -> datetime:
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


def next_day(value: datetime) -> datetime:
    '''Increment by one day'''
    day = 1 + value.day
    try:
        value = value.replace(day)
    except ValueError:
        day = 1
        value = value.replace(day)
        value = next_month(value)
    return value


class TimeIndicator:
    time_type = 'https://metadata.datadrivendiscovery.org/types/Time'
    integer_type = 'http://schema.org/Integer'
    categorical_type = 'https://metadata.datadrivendiscovery.org/types/CategoricalData'

    def __init__(self, training_data: container.DataFrame):
        self.time_indicator_index = training_data.get_columns_with_semantic_type(self.time_semantic_type)
        if len(self.time_indicator_index) > 1:
            _logger.warn(f'More than one time indicator columns. Using first one: {self.time_indicator_index}')
        self.time_indicator_index = self.time_indicator_index[0]
        self.indicator_style = self._deduce_style(training_data, self.time_indicator_index)

        self.year_index = -1
        if self.indicator_style == TimeIndicatorType.MONTH_DAY:
            self.year_index = self._find_year_index(training_data)

    def get_datetime(self, row: pd.Series) -> typing.UNION[int, datetime.date]:
        value = row.iloc[self.time_indicator_index]
        if self.indicator_style == TimeIndicatorType.INTEGER:
            return int(value)
        if self.indicator_style == TimeIndicatorType.YEAR:
            return datetime.date(int(value), 1, 1)
        fields = [int(x) for x in value.split('-')]
        if self.indicator_style == TimeIndicatorType.YEAR_MONTH:
            return datetime.date(fields[0], fields[1], 1)
        if self.indicator_style == TimeIndicatorType.MONTH_DAY:
            year = int(row.iloc[self.year_index])
            return datetime.date(year, fields[0], fields[1])

    def get_next_time(self, date: typing.UNION[int, datetime.date]) -> typing.UNION[int, datetime.date]:
        if self.indicator_style == TimeIndicatorType.INTEGER:
            return date+1
        if self.indicator_style == TimeIndicatorType.YEAR:
            return date.replace(year=date.year+1)
        if self.indicator_style == TimeIndicatorType.YEAR_MONTH:
            return next_month(date)
        if self.indicator_style == TimeIndicatorType.MONTH_DAY:
            return next_day(date)
        return None

    def get_difference(self, start: typing.UNION[int, datetime.date], end: typing.UNION[int, datetime.date]) -> int:
        if self.indicator_style == TimeIndicatorType.INTEGER:
            return end - start
        if self.indicator_style == TimeIndicatorType.YEAR:
            return end.year - start.year
        delta = end - start
        if self.indicator_style == TimeIndicatorType.YEAR_MONTH:
            return round(delta.days/30)
        if self.indicator_style == TimeIndicatorType.MONTH_DAY:
            return delta.days
        return None

    def _deduce_style(self, training_data: container.DataFrame, time_indicator_index: int) -> TimeIndicatorType:
        metadata = training_data.query([mbase.metadata, time_indicator_index])
        if self.time_type in metadata['semantic_types']:
            return TimeIndicatorType.INTEGER

        column = training_data.iloc[:, time_indicator_index]
        if len(column) > 50:
            subset_index = np.random.choice(len(column), 50, relpace=False)
        else:
            subset_index = range(len(column))
        values = column[subset_index]
        fields = [value.split('-') for value in values]

        field_counts = np.array([len(field) for field in fields])
        if np.all(field_counts == 1):
            # No divider '-' in any fields, i.e. one field
            return TimeIndicatorType.YEAR
        elif np.all(field_counts != 2):
            # More than two filelds
            return TimeIndicatorType.OTHER

        field0 = np.array([x[0] for x in fields])

        if np.any(field0 > 12):
            return TimeIndicatorType.YEAR_MONTH
        else:
            return TimeIndicatorType.MONTH_DAY

    def _find_year_index(self, training_data: container.DataFrame) -> int:
        categorical_indices = training_data.get_columns_with_semantic_type(self.categorical_type)
        for index in categorical_indices:
            metadata = training_data.query([mbase.metadata, index])
            if metadata['structural_type'] == 'str':
                if self._string_column_is_year(training_data.iloc[:, index]):
                    return index
            elif metadata['structural_type'] == 'int':
                if self._int_column_is_year(training_data.iloc[:, index]):
                    return index

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
        self.column_index = {}

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

# def is_year_month_format(column, threshold=0.05) -> bool:
#     if len(column) > 50:
#         subset_index = np.random.choice(len(column), 50)
#     else:
#         subset_index = range(len(column))
#     wrong_count = 0
#     for index in subset_index:
#         if isinstance(column[index], str):
#             fields = list(column[index].split('-'))
#             if len(fields) != 2:
#                 # allow year-month-day format
#                 wrong_count += 1
#             else:
#                 try:
#                     year = int(fields[0])
#                     month = int(fields[1])
#                     if year > 1 or month > 1 or month > 12:
#                         wrong_count += 1
#                 except Exception:
#                     wrong_count += 1
#         else:
#             wrong_count += 1
#     return wrong_count/len(subset_index) < threshold


# def year_month_to_int(column: typing.Sequence) -> typing.Sequence[int]:
#     return [int(datetime.date(dt).strftime('%s'))
#             if dt is not np.nan else np.nan
#             for dt in year_month_to_datetime(column)]


# def year_month_to_datetime(column: typing.Sequence) -> typing.Sequence[datetime]:
#     result = []
#     for date_str in column:
#         try:
#             fields = list(date_str.split('-'))
#             year = int(fields[0])
#             month = int(fields[1])
#             if len(fields) > 2:
#                 day = int(fields[2])
#             else:
#                 day = 1
#             result.append(datetime.date(year, month, day))
#         except Exception:
#             result.append(np.nan)
#     return result
