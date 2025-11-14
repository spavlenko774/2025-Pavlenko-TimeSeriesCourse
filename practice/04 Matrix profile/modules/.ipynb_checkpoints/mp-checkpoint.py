import numpy as np
import pandas as pd
import math

import stumpy
from stumpy import config


def compute_mp(ts1: np.ndarray, m: int, exclusion_zone: int = None, ts2: np.ndarray = None):
    """
    Compute the matrix profile

    Parameters
    ----------
    ts1: the first time series
    m: the subsequence length
    exclusion_zone: exclusion zone
    ts2: the second time series

    Returns
    -------
    output: the matrix profile structure
            (matrix profile, matrix profile index, subsequence length, exclusion zone, the first and second time series)
    """

    # stumpy ожидает float64
    ts1 = ts1.astype(np.float64)
    if ts2 is not None:
        ts2 = ts2.astype(np.float64)

    # если exclusion_zone не задан – дефолт
    if exclusion_zone is None:
        exclusion_zone = int(np.ceil(m / 4))

    # self-join (один ряд)
    if ts2 is None:
        mp = stumpy.stump(ts1, m)
    else:
        # AB-join (два разных ряда)
        # здесь ignore_trivial=False, чтобы НЕ отбрасывать "тривиальные" совпадения,
        # т.к. ряды разные
        mp = stumpy.stump(ts1, m, ts2, ignore_trivial=False)

    return {'mp': mp[:, 0],
            'mpi': mp[:, 1],
            'm' : m,
            'excl_zone': exclusion_zone,
            'data': {'ts1' : ts1, 'ts2' : ts2}
            }
