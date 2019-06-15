import datetime
import logging
import typing

import numpy as np
import pandas as pd

_logger = logging.getLogger()


def is_year_month_format(column, threshold=0.05) -> bool:
    if len(column) > 50:
        subset_index = np.random.choice(len(column), 50)
    else:
        subset_index = range(len(column))
    wrong_count = 0
    for index in subset_index:
        if isinstance(column[index], str):
            fields = list(column[index].split('-'))
            if (len(fields) < 2 or len(fields) > 3):
                # allow year-month-day format
                wrong_count += 1
            else:
                try:
                    for value in fields:
                        if int(value) == 0:
                            wrong_count += 1
                            continue
                except Exception:
                    wrong_count += 1
        else:
            wrong_count += 1
    return wrong_count/len(subset_index) < threshold


def year_month_to_int(column: typing.Sequence) -> typing.Sequence[int]:
    return [int(datetime.date(dt).strftime('%s'))
            if dt is not np.nan else np.nan
            for dt in year_month_to_datetime(column)]


def year_month_to_datetime(column: typing.Sequence) -> typing.Sequence[datetime]:
    result = []
    for date_str in column:
        try:
            fields = list(date_str.split('-'))
            year = int(fields[0])
            month = int(fields[1])
            if len(fields) > 2:
                day = int(fields[2])
            else:
                day = 1
            result.append(datetime.date(year, month, day))
        except Exception:
            result.append(np.nan)
    return result


def next_month(value: datetime) -> datetime:
    tuple = value.timetuple
    year = tuple.tm_year
    month = tuple.tm_mon
    if month == 12:
        year = year + 1
        month = 1
    else:
        month += 1

    # May fail if tuple.tm_mday > 28
    return value.replace(year=year, month=month)


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
        self.range = self.get_time_range(data.iloc[:, time_index])

        # Maps column tuple name to integer, e.g. ('cas9_CAD', 'S_0102') maps to 0.
        self.column_index = {}

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
        time_range = self.get_time_range(self.data.iloc[:, self.time_index])

        complete_index = pd.Index(time_range, name=self.data.columns[self.time_index])
        time_vectors = pd.DataFrame(index=complete_index, columns=column_names)
        for i, (name, group) in enumerate(self.data.groupby(categorical_column_name)):
            self.column_index[name] = i
            col = pd.DataFrame(group.iloc[:, self.value_index])
            col = col.set_index(group.iloc[:, self.time_index])
            col.reindex(complete_index)
            time_vectors.iloc[:, i] = col.iloc[:, 0]
        return time_vectors
