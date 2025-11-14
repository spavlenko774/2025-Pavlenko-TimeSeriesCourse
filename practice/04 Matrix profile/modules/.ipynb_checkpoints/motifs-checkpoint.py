import numpy as np

from modules.utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    mp = matrix_profile['mp'].copy()      # матричный профиль
    mpi = matrix_profile['mpi']           # индексы ближайших соседей
    excl_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        # индекс минимального значения (кандидат в мотив)
        i = int(np.argmin(mp))
        dist = mp[i]

        # если больше нет конечных значений — выходим
        if not np.isfinite(dist):
            break

        # индекс пары (второй участник мотива)
        j = int(mpi[i])

        # упорядочим индексы (левый, правый)
        left = min(i, j)
        right = max(i, j)

        motifs_idx.append((left, right))
        motifs_dist.append(dist)

        # исключаем окрестность вокруг обоих индексов
        mp = apply_exclusion_zone(mp, i, excl_zone, np.inf)
        mp = apply_exclusion_zone(mp, j, excl_zone, np.inf)

    return {
        "indices" : motifs_idx,
        "distances" : motifs_dist
        }
