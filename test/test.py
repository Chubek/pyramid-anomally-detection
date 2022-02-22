from algorithms.simple import distance_to_all_points, distance_to_nearest_neighbor, distance_to_k_nearest_neighbor
from algorithms.k_means import k_means_clustering
from algorithms.fuzzy_c_means import FuzzyCMeans
import numpy as np
from scipy.spatial.distance import euclidean
from copy import deepcopy
import matplotlib.pyplot as plt


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


def test_concept():
    a = [1, 2, 3]
    b = None

    aabb = {"a": a, "b": b}

    def inner():
        aabb['b'] = deepcopy(aabb['a'])
        aabb['a'] = []

    inner()
    print("\n", aabb)


def test_kmeans():
    dataset = np.array([1, 3, 5, 7, 100, 101, 200, 202, 205, 208, 210, 212, 214])

    distance_name = "euclidean"

    cent, cls = k_means_clustering(dataset, k=3, distance_name=distance_name)

    print("\n", cent, "\n", cls)


def test_gamma():
    df = np.array([(1, 3), (2, 5), (4, 8), (7, 9)])
    gamma = initialize_gamma(df, 3)

    cents = calculate_centroid(df, gamma, 2)

    print("\n", cents)


def test_cmeansfuzzy():
    dataset = np.array([1, 3, 5, 7, 100, 101, 200, 202, 205, 208, 210, 212, 214])

    distance_name = "euclidean"

    c = 3

    fuzzyc = FuzzyCMeans(dataset, c, distance_name)

    ret = fuzzyc()

    for j in range(len(ret)):
        plt.plot(dataset, ret[j, :], c=f"#{np.random.randint(0, 255)}00fc")

    plt.show()

    print(c)
