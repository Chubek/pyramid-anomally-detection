import numpy as np
from typing import Callable
from .distance import return_distance


class FuzzyCMeans:
    def __init__(self, df: np.array, c: int, dist_func_name: str, m=2.0, minowski_norm=2, tol=1e-3):
        self.df = df
        self.c = c
        self.dist_func = self.get_dist_func(dist_func_name, minowski_norm)
        self.m = m
        self.m2 = 2 / (m - 1)
        self.tol = tol
        self.u = self.initialize_gamma()
        self.u_prev = self.initialize_gamma()
        self.centroids = self.initialize_calc_centroids()
        self.ij_pair = np.array(np.sum([[(i, j) for i in range(len(df))] for j in range(c)], []))

    def get_dist_func(self, distance_name, minowski_norm):
        distance_func = return_distance(distance_name)

        match distance_name:
            case 'mahalanobis':
                covariance_matrix = np.cov(self.df)
                dist_lambda = lambda x, y: distance_func(x, y, covariance_matrix)
            case "minowski":
                dist_lambda = lambda x, y: distance_func(x, y, minowski_norm)
            case _:
                dist_lambda = lambda x, y: distance_func(x, y)

        return dist_lambda

    def initialize_gamma(self):
        return np.random.uniform(0, 1, [self.c, len(self.df)])

    def initialize_calc_centroids(self):
        gammas_ = self.u ** self.m

        cents = []

        def cent_calc(gamma_arg):
            sums = []

            calc = lambda x: sums.append(np.dot(gamma_arg, x))

            np.vectorize(calc, signature="(n)->()")(self.df)

            sum_sum = np.sum(np.array(sums), axis=-1)

            sum_div = sum_sum / np.sum(gammas_)

            cents.append(sum_div)

        np.vectorize(cent_calc, signature='(n)->()')(gammas_)

        return np.array(cents)

    def update_u(self, tup):
        i, j = tup
        cj = self.centroids[j]
        xi = self.df[i]

        cks = self.centroids[self.centroids != cj]

        x_ck = lambda ck: self.dist_func(xi, cj) / self.dist_func(xi, ck)

        res = np.vectorize(x_ck)(cks)

        res_op = 1 / res ** self.m2

        self.u[i, j] = res_op

    def iter(self):
        np.vectorize(self.update_u, signature="(n)->()")(self.ij_pair)

    def check_conv(self):
        return np.sum((self.u - self.u_prev) ** 2) <= self.tol

    def __call__(self, num_iter=1000):
        for i in range(num_iter):
            self.iter()

            if self.check_conv():
                break

            self.centroids = self.initialize_calc_centroids()

        return self.u
