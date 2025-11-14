import numpy as np

from modules.utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """
 
    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    mp = matrix_profile['mp'].copy()        # профиль (1D)
    mpi = matrix_profile['mpi']             # индексы ближайших соседей

    excl_zone = matrix_profile.get('excl_zone')
    if excl_zone is None:
        m = matrix_profile['m']
        excl_zone = int(np.ceil(m / 4))

    for _ in range(top_k):
        # ищем самый "аномальный" индекс — максимальное значение профиля
        i = int(np.nanargmax(mp))
        dist = mp[i]

        # если нормальных значений больше нет — выходим
        if not np.isfinite(dist) or dist == -np.inf:
            break

        nn = int(mpi[i])  # индекс ближайшего соседа

        discords_idx.append(i)
        discords_dist.append(dist)
        discords_nn_idx.append(nn)

        # выжигаем окрестность вокруг диссонанса и его соседа
        mp = apply_exclusion_zone(mp, i, excl_zone, -np.inf)
        mp = apply_exclusion_zone(mp, nn, excl_zone, -np.inf)

    return {
        'indices' : discords_idx,
        'distances' : discords_dist,
        'nn_indices' : discords_nn_idx
        }
