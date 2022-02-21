from algorithms.simple import distance_to_all_points, distance_to_nearest_neighbor
import numpy as np


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
