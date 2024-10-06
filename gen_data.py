import numpy as np
from sklearn.datasets import make_blobs

# Generate a Multimodal Dataset
def generate_multimodal_data(n_samples=1000, n_features=2, n_classes=3, n_clusters_per_class=3):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters_per_class * n_classes,
                      cluster_std=1.0, random_state=42)

    # Assign each cluster to a class
    y_mod = np.zeros_like(y)
    clusters_per_class = n_clusters_per_class
    for i in range(n_classes):
        y_mod[np.logical_and(y >= i * clusters_per_class, y < (i + 1) * clusters_per_class)] = i

    return X, y_mod
