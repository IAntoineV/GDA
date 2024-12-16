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

# Generate random means and stds for blobs
def generate_means_stds(n_components, n_features, mean_range=(-10, 10), std_range=(.5, 3.5)):
    means = np.random.uniform(mean_range[0], mean_range[1], size=(n_components, n_features))
    
    stds = np.random.uniform(std_range[0], std_range[1], size=(n_components, n_features))
    
    return means, stds