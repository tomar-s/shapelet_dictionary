import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ShapeletTransform(BaseEstimator, TransformerMixin):
    def __init__(self, shapelets, normalize=True):
        self.shapelets = shapelets
        self.normalize = normalize

    def _z_normalize(self, x):
        return (x - np.mean(x)) / (np.std(x) + 1e-8)

    def transform(self, X):
        n_samples, series_len = X.shape
        shapelet_len = self.shapelets[0].shape[0]
        n_shapelets = len(self.shapelets)

        X_transformed = np.zeros((n_samples, n_shapelets))

        for i in range(n_samples):
            series = X[i]
            # if self.normalize:
            #     series = self._z_normalize(series)

            for j, shapelet in enumerate(self.shapelets):
                min_dist = float("inf")
                for start in range(series_len - shapelet_len + 1):
                    subseq = series[start:start + shapelet_len]
                    if self.normalize:
                        subseq = self._z_normalize(subseq)
                        shapelet = self._z_normalize(shapelet)
                    dist = np.linalg.norm(subseq - shapelet)
                    min_dist = min(min_dist, dist)

                X_transformed[i, j] = min_dist

        return X_transformed

    def fit(self, X, y=None):
        return self  # stateless