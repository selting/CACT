import datetime as dt

import numpy as np
import pandas as pd


def round_timedelta(td: dt.timedelta, resolution: str):
    """

    :param td: timedelta value
    :param resolution: 's' for seconds, 'm' for minutes, 'h' for hours'
    :return:
    """
    assert td >= dt.timedelta(0), "Not implemented for negative values"
    if resolution == "s":
        res = dt.timedelta(seconds=round(td.total_seconds(), None))
    elif resolution == "m":
        res = dt.timedelta(minutes=(int(td.total_seconds() // 60)))
    elif resolution == "h":
        res = dt.timedelta(minutes=(int(td.total_seconds() // 60**2)))
    else:
        raise ValueError()
    return res


def ceil_timedelta(td: dt.timedelta, resolution: str):
    """

    :param td:
    :param resolution: 's' for seconds, 'm' for minutes, 'h' for hours'
    :return:
    """
    assert td >= dt.timedelta(0), "Not implemented for negative values"
    a = td.total_seconds()
    if resolution == "s":
        b = 1
        x = a // b + bool(a % b)
        res = dt.timedelta(seconds=x)
    elif resolution == "m":
        b = 60
        x = a // b + bool(a % b)
        res = dt.timedelta(minutes=x)
    elif resolution == "h":
        b = 60**2
        x = a // b + bool(a % b)
        res = dt.timedelta(hours=x)
    else:
        raise ValueError()

    return res


total_seconds_vectorized = np.vectorize(lambda x: x.total_seconds(), otypes=[float])


def uniform_sample_datetime(start: dt.datetime, end: dt.datetime, size: int = 1):
    """
    Uniformly sample datetime objects between start and end
    :param start:
    :param end:
    :param size:
    :return:
    """
    assert start < end, "Invalid start and end values"
    delta = end - start
    samples = np.random.uniform(0, delta.total_seconds(), size)
    samples = [start + dt.timedelta(seconds=ts) for ts in samples]
    if size > 1:
        return samples
    else:
        return samples[0]


def safe_divide_timedelta(a, b, default_value=None):
    """

    Parameters
    ----------
    a: pd.Series
    b: pd.Series
    default_value: value to return if b is 0 or timedelta(0)

    Returns
    -------

    """
    results = []
    for a_val, b_val in zip(a, b):
        if b_val == 0 or b_val == dt.timedelta(0):
            result = default_value
        else:
            result = a_val / b_val
        results.append(result)
    return pd.Series(results, index=a.index)
