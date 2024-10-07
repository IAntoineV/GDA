import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-3):
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

    def _initialize_parameters(self, X):
        """
        Initialize the GMM parameters: weights, means, and covariances.

        Parameters:
        - X: Input data of shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = X[np.random.choice(n_samples, self.n_components, False)]
        self.covariances_ = np.array([np.cov(X.T) for _ in range(self.n_components)])

    def _e_step(self, X):
        """
        E-step: Compute the responsibilities for each Gaussian component.

        Parameters:
        - X: Input data of shape (n_samples, n_features).

        Returns:
        - responsibilities: An array of shape (n_samples, n_components) representing the probability
          that each data point belongs to each Gaussian component.
        """
        n_samples, n_features = X.shape
        likelihoods = np.zeros((n_samples, self.n_components))

        # Compute the likelihood of each data point under each Gaussian component
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights_[k] * multivariate_normal.pdf(
                X, mean=self.means_[k], cov=self.covariances_[k]
            )

        # Compute responsibilities (posterior probabilities)
        total_likelihood = np.sum(likelihoods, axis=1, keepdims=True)
        responsibilities = likelihoods / total_likelihood
        return responsibilities

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
            responsibilities = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            # Check for convergence
            new_log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
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
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """
        Predict the posterior probabilities of the Gaussian components for each data point.

        Parameters:
        - X: Input data of shape (n_samples, n_features).

        Returns:
        - responsibilities: An array of shape (n_samples, n_components) representing the posterior probabilities.
        """
        return self._e_step(X)

