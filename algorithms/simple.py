from .distance import return_distance
import numpy as np
import pandas as pd
from typing import Any, Iterable, Callable


def distance_to_all_points(dataset: np.array, distance_name: str, minowski_norm=2) -> pd.DataFrame and Any:
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

    return ret_df, ret_df.iloc[ret_df.idxmax(), :]


def distance_to_nearest_neighbor(dataset: np.array, distance_name: str, minowski_norm=2) -> pd.DataFrame and Any:
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

        y_ret = np.vectorize(lambda_x)(y[y != x])

        mapped_values.append((x, np.min(y_ret)))

    lambda_y = lambda x: func_map(x, dataset)

    np.vectorize(lambda_y)(dataset)

    ret_df = pd.DataFrame.from_records(mapped_values, columns=["Metric", "NN_Distance"])

    return ret_df, ret_df.iloc[ret_df.idxmax(), :]


def distance_to_k_nearest_neighbor(dataset: np.array, distance_name: str, k=3, op="mean", minowski_norm=2) -> pd.DataFrame and Any:
    if k >= dataset.shape[0]:
        raise ValueError("Length of array is smaller than, or equal to, k!")

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

    def func_map(x: Iterable, operator=op):
        first_val = x[0]
        rest = x[1:]
        dist_calc = lambda y: dist_lambda(first_val, y)
        res = np.vectorize(dist_calc)(rest)
        match operator:
            case "sum":
                op_res = np.sum(res)
            case "mean":
                op_res = np.mean(res)
            case "median":
                op_res = np.median(res)
            case _:
                is_inst = isinstance(operator, Callable)
                match is_inst:
                    case True:
                        op_res = operator(res)
                    case False:
                        raise ValueError("`op` can only be: sum, mean, median or a callable with signature (np.array) "
                                         "-> float")

        mapped_values.append((first_val, op_res))

    list_ds = dataset.tolist()

    list_ds = list_ds + [0 for _ in range(k - (len(list_ds) % k))]

    zip_list = []

    for i in range(k + 1):
        zip_list.append(list_ds[i:-(k - i + 1)])

    zip_ds = np.array(list(zip(*zip_list)))

    np.vectorize(func_map, signature="(n)->()")(zip_ds)

    ret_df = pd.DataFrame.from_records(mapped_values, columns=["Metric", "KNN_Distance"])

    return ret_df, ret_df.iloc[ret_df.idxmax(), :]
