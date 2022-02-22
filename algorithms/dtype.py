import numpy as np


class KMeansHolder:
    centroid: np.array
    corr_cluster: np.array

    def __init__(self, centroid):
        self.centroid = centroid
        self.corr_cluster = np.array([])

    def update_means(self):
        self.centroid = np.mean(self.corr_cluster)
