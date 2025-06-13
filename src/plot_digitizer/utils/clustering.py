from typing import Iterable, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_centroids(points: np.ndarray, eps: float = 4.0, min_samples: int = 5) -> List[Tuple[float, float]]:
    """Utility wrapper around DBSCAN that returns cluster centroids."""
    if len(points) == 0:
        return []
    cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return [
        points[cluster.labels_ == cid].mean(axis=0)
        for cid in set(cluster.labels_)
        if cid != -1
    ]
