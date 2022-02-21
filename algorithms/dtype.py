import numpy as np


class KMeansHolder:
    centroid: np.array
    corr_cluster: np.array

    def __init__(self, centroid, corr_cluster):
        self.centroid = centroid
        self.corr_cluster = corr_cluster

    def update_means(self):
        self.centroid = np.mean(self.corr_cluster, axis=-1)
