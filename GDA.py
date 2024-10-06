import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA

# GMM Discriminant Analysis
class GDA:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.gmms = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.gmms = {}

        # Fit GMM to each class
        for cls in self.classes_:
            gmm = GaussianMixture(n_components=self.n_components, covariance_type='full')
            gmm.fit(X[y == cls])
            self.gmms[cls] = gmm

    def predict(self, X):
        log_likelihoods = np.zeros((X.shape[0], len(self.classes_)))

        # Compute log-likelihood for each sample under each class GMM
        for i, cls in enumerate(self.classes_):
            log_likelihoods[:, i] = self.gmms[cls].score_samples(X)

        # Predict the class with the highest log-likelihood
        return self.classes_[np.argmax(log_likelihoods, axis=1)]

    def predict_proba(self, X):
        log_likelihoods = np.zeros((X.shape[0], len(self.classes_)))

        # Compute log-likelihood for each sample under each class GMM
        for i, cls in enumerate(self.classes_):
            log_likelihoods[:, i] = self.gmms[cls].score_samples(X)

        # Convert log-likelihoods to probabilities using softmax
        likelihoods = np.exp(log_likelihoods)
        return likelihoods / np.sum(likelihoods, axis=1, keepdims=True)

