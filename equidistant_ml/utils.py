"""Utils."""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


np.random.seed(42)


def sget(dct: dict, *keys):
    """Safe version of get.

    example usage:
        my_dict = {'a': [{'b': {'c': 'my_val'} } ] }

        sget(my_dict, 'a', 0, 'b', 'c')  # returns 'my_val'
        sget(my_dict, 'a', 0, 'b', 'c', 'd')  # returns None
    """
    dct = dct.copy()
    for key in keys:
        try:
            dct = dct[key]
        except (KeyError, IndexError):
            return None
    return dct


class DatetimeUtils:
    @staticmethod
    def str_time_prop(start, end, time_format, prop):
        """Get a time at a proportion of a range of two formatted times.

        start and end should be strings specifying times formatted in the
        given format (strftime-style), giving an interval [start, end].
        prop specifies how a proportion of the interval to be taken after
        start.  The returned time will be in the specified format.
        """

        stime = time.mktime(time.strptime(start, time_format))
        etime = time.mktime(time.strptime(end, time_format))

        ptime = stime + prop * (etime - stime)

        return time.strftime(time_format, time.localtime(ptime))

    @classmethod
    def random_date(cls, start, end, prop):
        return cls.str_time_prop(start, end, '"%Y-%m-%dT%H:%M:%S', prop)

    @staticmethod
    def random_date_distributed(year=2021) -> str:
        """Get a random datetime but weighted towards waking hours.

        # visualise the distribution
        # pd.concat((
        #     (pd.Series(np.random.normal(9, 2, 5000) % 24)),  # the morning rush hour
        #     (pd.Series(np.random.normal(13, 5, 5000) % 24)),  # the general traffic
        #     (pd.Series(np.random.normal(19, 4, 10000) % 24))  # the evening rush
        #     )).plot.density()
        """

        pad = lambda x: str(x).zfill(2)

        # generate a date up to 50 days in the future (when google stops giving directions)
        days_from_now = np.random.randint(1, 50)
        new_datetime = datetime.now() + timedelta(days=days_from_now)

        month = pad(new_datetime.month)
        day = pad(new_datetime.day)
        hour = pad(int(np.random.choice(np.concatenate([
            (np.random.normal(9, 2, 1) % 24),
            (np.random.normal(13, 5, 1) % 24),
            (np.random.normal(19, 4, 2) % 24)
            ]), 1, replace=True)))

        date = f"{year}-{month}-{day}T{hour}:00:00"

        return date

    @staticmethod
    def datestr_to_epoch(date: str, format_: str = "%Y-%m-%dT%H:%M:%S") -> float:
        time_ = datetime.strptime(date, format_)
        epoch_time = (time_ - datetime(1970, 1, 1)).total_seconds()
        return epoch_time

    @classmethod
    def get_random_epoch_time(cls) -> str:
        return str(int(cls.datestr_to_epoch(cls.random_date_distributed())))
