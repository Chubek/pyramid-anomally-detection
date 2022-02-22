import numpy as np
from os import urandom
from .dtype import KMeansHolder
from typing import List, Callable, Tuple
from .distance import return_distance
from copy import deepcopy


def select_random_centroids(df: np.array, k: int) -> np.array:
    np.random.seed(int("".join([str(int(c)) for c in urandom(2)])))

    centroids = df[np.random.randint(0, len(df), k)]

    return centroids


def get_nearest_index(points: List[np.array], target: np.array, dist_func: Callable):
    dist_target = lambda source: dist_func(source, target)

    distances = np.vectorize(dist_target)(points)

    return np.argmin(distances)


def check_tolerance(prev: List[KMeansHolder], current: List[KMeansHolder], tol: float):
    prev_ = np.array([pr.centroid for pr in prev])
    curr_ = np.array([curr.centroid for curr in current])

    return np.sum((prev_ - curr_) ** 2) <= tol


def k_means_clustering(df: np.array, k: int, distance_name: str, num_iter=1000,
                                                  minowski_norm=2, tol=1e-3) -> Tuple[List, List]:

    distance_func = return_distance(distance_name)

    match distance_name:
        case 'mahalanobis':
            covariance_matrix = np.cov(df)
            dist_lambda = lambda x, y: distance_func(x, y, covariance_matrix)
        case "minowski":
            dist_lambda = lambda x, y: distance_func(x, y, minowski_norm)
        case _:
            dist_lambda = lambda x, y: distance_func(x, y)

    centroids = select_random_centroids(df, k)
    clusters_current = [KMeansHolder(centroid=c, corr_cluster=np.array([])) for c in centroids]
    clusters_prev = [KMeansHolder(centroid=c, corr_cluster=np.array([])) for c in centroids]

    head = {
        "clusters_current": clusters_current,
        "clusters_prev": clusters_prev,
        "dist_func": dist_lambda
    }

    def get_nearest_cluster(point: float):
        points = [cls.centroid for cls in head['clusters_current']]

        argmin_ = get_nearest_index(points, point, head['dist_func'])

        head['clusters_current'][argmin_].corr_cluster = np.append(
            head['clusters_current'][argmin_].corr_cluster, point)
        head['clusters_current'][argmin_].update_means()

    def reset_clusters():
        head['clusters_prev'] = deepcopy(head['clusters_current'])

        for i in range(len(head['clusters_current'])):
            head["clusters_current"][i].corr_cluster = np.array([])

    def check_tol():
        return check_tolerance(head['clusters_prev'], head["clusters_current"], tol)

    for i in range(num_iter):
        np.vectorize(get_nearest_cluster)(df)

        if i > num_iter // 10:
            if check_tol():
                break

        reset_clusters()

    return [c.centroid.tolist() for c in head['clusters_prev']], [c.corr_cluster.tolist() for c in
                                                                  head['clusters_prev']]


def initialize_gamma(df: np.array, c: int):
    return np.random.uniform(0, 1, [c, df.shape[0]])


def calculate_centroid(df: np.array, gammas: np.array, m: float):
    gammas_ = gammas + m

    cents = []

    def cent_calc(x):
        lam_calc = lambda x_: (x_ * gammas_) / np.sum(gammas_)

        res = np.vectorize(lam_calc)(x)

        cents.append(res)

    np.vectorize(cent_calc)(df)

    return np.array(cents)


