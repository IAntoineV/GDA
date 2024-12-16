from typing import Type

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import t
from scipy.special import digamma, gamma
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



class GMM:
    def __init__(
        self, n_components, max_iter=100, tol=1e-3, km_init=False, km_cov_init=False,
    covariance_type="full"):
        """
        Gaussian Mixture Model using the Expectation-Maximization algorithm.

        Parameters:
        - n_components: The number of Gaussian components.
        - max_iter: Maximum number of iterations for the EM algorithm.
        - tol: Tolerance to declare convergence based on the log-likelihood change.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.kmeans_init = km_init
        self.kmeans_covariance_init = km_cov_init
        self.eps = 1e-5
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
            ) + self.eps
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
        eps = 1e-4
        likelihoods = self._likelihoods(X)
        # Compute responsibilities (posterior probabilities)
        total_likelihood = np.sum(likelihoods, axis=1, keepdims=True) + eps
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
        if self.covariance_type == "full":
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_sum = np.dot(responsibilities[:, k] * diff.T, diff)
                self.covariances_[k] = weighted_sum / Nk[k]
        elif self.covariance_type == "diag":
            for k in range(self.n_components):
                diff = X - self.means_[k]
                weighted_sum = np.dot(responsibilities[:, k] * diff.T, diff)
                self.covariances_[k] = np.diag(np.diag(weighted_sum)) / Nk[k]
        elif self.covariance_type == "tied":
            diff = X - self.means_[0]
            weighted_sum = np.dot(responsibilities[:, 0] * diff.T, diff)
            for k in range(self.n_components):
                self.covariances_[k] = weighted_sum / Nk[k]
        elif self.covariance_type == "spherical":
            diff = X - self.means_[0]
            weighted_sum = np.dot(responsibilities[:, 0] * diff.T, diff)
            for k in range(self.n_components):
                self.covariances_[k] = np.diag(np.diag(weighted_sum)) / Nk[k]

    def fit(self, X, verbose=False):
        """
        Fit the GMM model to the data using the EM algorithm.

        Parameters:
        - X: Input data of shape (n_samples, n_features).
        """
        self._initialize_parameters(X)
        log_likelihood = 0

        for i in range(self.max_iter):
            if verbose:
                print(f'{ int(1000 * i / self.max_iter)/10 }% fit')
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
        covariance_type="full",
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
        self.covariance_type = covariance_type
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
                    covariance_type=self.covariance_type,
                )
                gmm.fit(X)
                running_BICs.append(self.bic(gmm, X))
            self.BICs[k - 1] = np.mean(running_BICs)

        k_opt = np.argmax(self.BICs) + 1

        self.gmm = GMM(
            n_components=k_opt,
            max_iter=self.max_iter,
            tol=self.tol,
            km_init=self.kmeans_init,
            km_cov_init=self.kmeans_covariance_init,
        )
        self.gmm.fit(X)

        return self.gmm, k_opt, self.BICs
    
    def score_samples(self, X):
        return np.log(np.sum(self.gmm._likelihoods(X), axis=1))

class SMM:
    def __init__(self, n_components, degrees_of_freedom, tol=1e-6, max_iter=100):
        """
        Initialize the Student's t-mixture model.

        Parameters:
        - n_components: int, number of mixture components
        - degrees_of_freedom: float, degrees of freedom for the t-distribution
        - tol: float, tolerance for convergence
        - max_iter: int, maximum number of iterations
        """
        self.n_components = n_components
        self.nu = degrees_of_freedom
        self.tol = tol
        self.max_iter = max_iter

        # Model parameters
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.eps = 1e-5

    def _initialize_params(self, X):
        """Randomly initialize mixture parameters."""
        n_samples, n_features = X.shape
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.cov(X, rowvar=False)] * self.n_components)
        
    def _likelihoods(self, X):
        n_samples, _ = X.shape
        likelihoods = np.zeros((n_samples, self.n_components))

        # Compute the likelihood of each data point under each Gaussian component
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights_[k] * multivariate_normal.pdf(
                X, mean=self.means_[k], cov=self.covariances_[k], allow_singular=True
            ) + self.eps
        return likelihoods

    def _e_step(self, X):
        """Perform the Expectation step."""
        n_samples, _ = X.shape
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            diff = X - self.means_[k]
            inv_cov = np.linalg.inv(self.covariances_[k])
            quad_form = np.sum(diff @ inv_cov * diff, axis=1)

            const = gamma((self.nu + X.shape[1]) / 2) / (
                gamma(self.nu / 2) * (self.nu * np.pi) ** (X.shape[1] / 2) * np.sqrt(np.linalg.det(self.covariances_[k]))
            )
            responsibilities[:, k] = self.weights_[k] * const * (1 + quad_form / self.nu) ** (-(self.nu + X.shape[1]) / 2)

        # Normalize responsibilities
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X, responsibilities):
        """Perform the Maximization step."""
        n_samples, n_features = X.shape

        # Effective number of points for each component
        Nk = responsibilities.sum(axis=0)

        # Update weights
        self.weights_ = Nk / n_samples

        # Update means
        self.means_ = np.array([
            (responsibilities[:, k, None] * X).sum(axis=0) / Nk[k]
            for k in range(self.n_components)
        ])

        # Update covariances
        self.covariances_ = np.array([
            np.cov(X - self.means_[k], rowvar=False, aweights=responsibilities[:, k])
            for k in range(self.n_components)
        ])

    def fit(self, X):
        """
        Fit the model to the data using the EM algorithm.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), input data
        """
        n_samples, n_features = X.shape
        self._initialize_params(X)

        log_likelihood = -np.inf
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            # Compute log-likelihood
            new_log_likelihood = np.sum(
                np.log(np.sum(responsibilities, axis=1))
            )

            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                print(f"Converged at iteration {iteration}.")
                break

            log_likelihood = new_log_likelihood

    def predict_proba(self, X):
        """
        Predict the posterior probabilities for each component.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), input data

        Returns:
        - responsibilities: ndarray of shape (n_samples, n_components)
        """
        return self._e_step(X)

    def predict(self, X):
        """
        Predict the most likely component for each sample.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), input data

        Returns:
        - labels: ndarray of shape (n_samples,), predicted component labels
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X):
        return np.log(np.sum(self._likelihoods(X), axis=1))