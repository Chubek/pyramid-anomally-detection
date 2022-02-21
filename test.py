from algorithms.simple import distance_to_all_points, distance_to_nearest_neighbor, distance_to_k_nearest_neighbor
import numpy as np
from scipy.spatial.distance import euclidean


def test_knn():
    arr = np.arange(0, 4)

    # 3 3 0

    k_odd = 3
    k_ev = 4

    mmm_ev = len(arr) % k_ev
    mmm_odd = len(arr) % k_odd

    print("\n", len(arr), mmm_ev, mmm_odd)
    """
      ls = arr.tolist()
    zip_list = []
    for i in range(k + 1):
        zip_list.append(ls[i:-(k - i + 1)])

    zip_ds = np.array(list(zip(*zip_list)))
    """


def test_distances_to_all_points():
    dataset = np.arange(0, 20)

    distance_name = "mahalanobis"

    return_df, max_ = distance_to_all_points(dataset, distance_name)

    print(max)
    print(return_df.head())


def test_distances_to_nearest_neighbor():
    dataset = np.array([1, 2, 3, 8, 20, 21])

    distance_name = "euclidean"

    return_df, max_ = distance_to_nearest_neighbor(dataset, distance_name)

    print(max_)
    print(return_df.head())


def test_distances_to_nearest_k_neighbors():
    dataset = np.array([1, 3, 5, 7, 100, 101, 200, 202, 205, 208, 210, 212, 214])

    distance_name = "euclidean"

    return_df, max_ = distance_to_k_nearest_neighbor(dataset, distance_name, k=3, op=lambda x: np.mean(x))

    print(max_)
    print(return_df.head())
