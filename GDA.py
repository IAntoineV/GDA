import numpy as np
from GMMs import GMM, BIC_GMM
from joblib import Parallel, delayed


# GMM Discriminant Analysis
class GDA:
    def __init__(self, n_components=1, use_bic=False, **GMM_kwargs):
        self.n_components = n_components
        self.gmms = {}
        self.classes_ = None
        self.use_bic = use_bic
        self.GMM_kwargs = GMM_kwargs

    def fit_class(self, X, y, cls):
        if isinstance(self.n_components, list):
            gmm_components = self.n_components[cls]
        else:
            gmm_components = self.n_components

        if self.use_bic:
            gmm = BIC_GMM(max_components=gmm_components, **self.GMM_kwargs)
            return gmm.fit(X[y == cls])[0]
        else:
            gmm = GMM(n_components=gmm_components, **self.GMM_kwargs)
            gmm.fit(X[y == cls])
            return gmm
    def fit(self, X, y, n_jobs=1):
        self.classes_ = np.unique(y)
        self.gmms = {}

        # Parallel Fit GMM to each class
        gmms = Parallel(n_jobs=n_jobs)(
            delayed(self.fit_class)(X, y, cls) for cls in self.classes_
        )

        for k, cls in enumerate(self.classes_):
            self.gmms[cls] = gmms[k]

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
