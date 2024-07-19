import pandas as pd
import numpy as np

class KNN:

    def EuclideanDistance(self, data_testing, data_training, X):
        # Mengambil fitur yang ada dalam DataFrame
        features = X.columns
        # Menghitung jarak Euclidean
        distance = np.linalg.norm(data_testing[features] - data_training[features])
        return distance
