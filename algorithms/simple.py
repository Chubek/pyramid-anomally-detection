from .distance import return_distance
import numpy as np
import pandas as pd


def distance_to_all_points(dataset: np.array, distance_name: str, minowski_norm=2) -> pd.DataFrame:
    distance_func = return_distance(distance_name)

    match distance_name:
        case 'mahalanobis':
            covariance_matrix = np.cov(dataset)
            dist_lambda = lambda x, y: distance_func(x, y, covariance_matrix)
        case "minowski":
            dist_lambda = lambda x, y: distance_func(x, y, minowski_norm)
        case _:
            dist_lambda = lambda x, y: distance_func(x, y)

    mapped_values = []

    def func_map(x: float, y: np.array):
        lambda_x = lambda y_: dist_lambda(x, y_)

        y_ret = np.vectorize(lambda_x)(y)

        mapped_values.append((x, np.sum(y_ret)))

    lambda_y = lambda x: func_map(x, dataset)

    np.vectorize(lambda_y)(dataset)

    ret_df = pd.DataFrame.from_records(mapped_values, columns=["Metric", "Sum_of_Distances"])

    return ret_df
