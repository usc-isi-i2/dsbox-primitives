import datetime
import typing

import numpy as np


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
