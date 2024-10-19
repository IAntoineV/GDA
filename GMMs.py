from typing import Type

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class GMM:
    def __init__(
        self, n_components, max_iter=100, tol=1e-3, km_init=False, km_cov_init=False
    ):
        """
        Gaussian Mixture Model using the Expectation-Maximization algorithm.

        Parameters:
        - n_components: The number of Gaussian components.
        - max_iter: Maximum number of iterations for the EM algorithm.
        - tol: Tolerance to declare convergence based on the log-likelihood change.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.kmeans_init = km_init
        self.kmeans_covariance_init = km_cov_init

    def _initialize_parameters(self, X, eps=1e-10):
        """
        Initialize the GMM parameters: weights, means, and covariances.

        Parameters:
        - X: Input data of shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape
        self.weights_ = np.ones(self.n_components) / self.n_components

        if self.kmeans_init:
            kmeans = KMeans(n_clusters=self.n_components, n_init=10)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        else:
            self.means_ = X[np.random.choice(n_samples, self.n_components, False)]

        if self.kmeans_init and self.kmeans_covariance_init:
            covariances_list_ = []
            for k in range(self.n_components):
                cluster_k = X[kmeans.labels_ == k]
                if cluster_k.shape[0] > 1:
                    covariances_list_.append(np.cov(cluster_k.T))
                else:
                    covariances_list_.append(np.eye(n_features) * 1e-6)
            self.covariances_ = np.array(covariances_list_)
        else:
            self.covariances_ = np.array(
                [np.cov(X.T) + eps * np.eye(X.shape[1]) for _ in range(self.n_components)]
            )

    def _likelihoods(self, X):
        n_samples, _ = X.shape
        likelihoods = np.zeros((n_samples, self.n_components))

        # Compute the likelihood of each data point under each Gaussian component
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights_[k] * multivariate_normal.pdf(
                X, mean=self.means_[k], cov=self.covariances_[k], allow_singular=True
            )
        return likelihoods

    def _e_step(self, X):
        """
        E-step: Compute the responsibilities for each Gaussian component.

        Parameters:
        - X: Input data of shape (n_samples, n_features).

        Returns:
        - responsibilities: An array of shape (n_samples, n_components) representing the probability
          that each data point belongs to each Gaussian component.
        - total_likelihoods: An array of shape (n_samples, 1) representing the likelihood of each x_i
          given the paraemeters.
        """
        n_samples, n_features = X.shape
        likelihoods = self._likelihoods(X)
        # Compute responsibilities (posterior probabilities)
        total_likelihood = np.sum(likelihoods, axis=1, keepdims=True)
        responsibilities = likelihoods / total_likelihood
        return responsibilities, total_likelihood

    def _m_step(self, X, responsibilities):
        """
        M-step: Update the parameters (weights, means, and covariances).

        Parameters:
        - X: Input data of shape (n_samples, n_features).
        - responsibilities: An array of shape (n_samples, n_components) representing the probability
          that each data point belongs to each Gaussian component.
        """
        n_samples, n_features = X.shape

        # Total responsibility assigned to each component
        Nk = np.sum(responsibilities, axis=0)

        # Update weights
        self.weights_ = Nk / n_samples

        # Update means
        self.means_ = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]

        # Update covariances
        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_sum = np.dot(responsibilities[:, k] * diff.T, diff)
            self.covariances_[k] = weighted_sum / Nk[k]

    def fit(self, X):
        """
        Fit the GMM model to the data using the EM algorithm.

        Parameters:
        - X: Input data of shape (n_samples, n_features).
        """
        self._initialize_parameters(X)
        log_likelihood = 0

        for i in range(self.max_iter):
            # E-step
            responsibilities, total_likelihood = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            # Check for convergence
            new_log_likelihood = np.sum(np.log(total_likelihood))

            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

    def predict(self, X):
        """
        Predict the Gaussian component for each data point.

        Parameters:
        - X: Input data of shape (n_samples, n_features).

        Returns:
        - labels: An array of shape (n_samples,) representing the predicted Gaussian component for each data point.
        """
        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """
        Predict the posterior probabilities of the Gaussian components for each data point.

        Parameters:
        - X: Input data of shape (n_samples, n_features).

        Returns:
        - responsibilities: An array of shape (n_samples, n_components) representing the posterior probabilities.
        """
        return self._e_step(X)[0]

    def score_samples(self, X):
        return np.log(np.sum(self._likelihoods(X), axis=1))


class BIC_GMM:
    def __init__(
        self,
        max_components,
        n_fits=10,
        max_iter=100,
        tol=1e-3,
        km_init=False,
        km_cov_init=False,
    ):
        """
        Gaussian Mixture Model using the Expectation-Maximization algorithm.

        Parameters:
        - max_components: The maximum number of Gaussian components.
        - n_fits: The number of GMM fits for each GMM parametrization.
        - max_iter: Maximum number of iterations for each EM algorithm.
        - tol: Tolerance to declare convergence based on the log-likelihood change.
        """
        self.max_components = max_components
        self.n_fits = n_fits
        self.max_iter = max_iter
        self.tol = tol
        self.kmeans_init = km_init
        self.kmeans_covariance_init = km_cov_init
        self.BICs = np.zeros(max_components)

    def bic(self, gmm_model: Type[GMM], X):
        """
        Computes BIC(GMM) = L(x|pi, theta) - (Mk/2)log(n)
        BIC criterion to maximize.
        """
        log_lh = np.sum(np.log(gmm_model._e_step(X)[1]))
        k = gmm_model.n_components
        n, d = X.shape
        Mk = k - 1 + k * d + k * d * (d + 1) / 2

        return log_lh - np.log(n) * Mk / 2

    def plot_BICs(self):
        ks = list(range(1, self.max_components + 1))

        plt.figure(figsize=(8, 6))
        plt.plot(ks, self.BICs, marker="^")

        plt.xlabel("Number of components")
        plt.ylabel("BIC")
        plt.show()

    def fit(self, X):
        """
        Apply BIC criterion to find the best parameter k for GMM

        Parameters:
        - X: Input data of shape (n_samples, n_features).
        """

        for k in range(1, self.max_components + 1):
            running_BICs = []
            for _ in range(self.n_fits):
                gmm = GMM(
                    n_components=k,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    km_init=self.kmeans_init,
                    km_cov_init=self.kmeans_covariance_init,
                )
                gmm.fit(X)
                running_BICs.append(self.bic(gmm, X))
            self.BICs[k - 1] = np.mean(running_BICs)

        k_opt = np.argmax(self.BICs) + 1

        gmm = GMM(
            n_components=k_opt,
            max_iter=self.max_iter,
            tol=self.tol,
            km_init=self.kmeans_init,
            km_cov_init=self.kmeans_covariance_init,
        )
        gmm.fit(X)

        return gmm, k_opt, self.BICs
