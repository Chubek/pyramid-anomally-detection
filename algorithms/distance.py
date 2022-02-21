from scipy.spatial.distance import  euclidean, minkowski, mahalanobis, cosine, jaccard
from typing import Callable


def return_distance(name: str) -> Callable:
    match name:
        case "euclidean":
            return euclidean
        case "minowski":
            return minkowski
        case "mahalanobis":
            return mahalanobis
        case "cosine":
            return cosine
        case "jaccard":
            return jaccard
        case _:
            raise ValueError("Wrong distance name!")
