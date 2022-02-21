from algorithms.simple import distance_to_all_points
import numpy as np


def test_distances_to_all_points():
    dataset = np.arange(0, 20)

    distance_name = "mahalanobis"

    return_df = distance_to_all_points(dataset, distance_name)

    print(return_df.head())
