from pydantic import BaseModel
from typing import List
import numpy as np


class KMeansHolder(BaseModel):
    centroid: np.array
    corr_cluster: np.array

    def update_means(self):
        self.centroid = np.mean(self.corr_cluster, axis=-1)



