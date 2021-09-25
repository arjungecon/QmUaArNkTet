import numpy as np
from GaussHermiteQuadrature import GaussHermiteQuadrature
from numpy.linalg import cholesky as chol
from numpy.random import normal as normal

# Frequently used commands
inv, ax, det = np.linalg.inv, np.newaxis, np.linalg.det
cos, pi, arccos, log = np.cos, np.pi, np.arccos, np.log


class ExpectationMultivariateNormal:

    """
        Class used to evaluate an integral corresponding to the expectation of a function under a multivariate
        normal distribution either using Gauss-Hermite quadrature or Monte-Carlo simulations.
    """

    def __init__(self,
                 func,
                 dim: np.int32,
                 normal_dist: dict):

        """
            :param func: The (multivariate) function over which the expectation is computed, function
            :param dim: The number of inputs in the function, integer
            :param normal_dist: Mean vector and covariance matrix of the multivariate normal distribution, dict
        """

        # Defining characteristics for the object used to run the integration procedures.
        self.D, self.f = dim, func

        self.mu, self.cov = np.reshape(normal_dist['Mean'], (1, dim)), normal_dist['Cov']

        self.integral_GH, self.integral_MC = None, None

    def evaluate_gauss_hermite(self,
                               nodes: np.int16):

        """
            Evaluates the expectation operator with respect to the input multivariate normal distribution using
            Gauss-Hermite quadrature.

            :param nodes: The number of nodes required to run the Gauss-Hermite quadrature, integer
        """

        # Initializing the Gauss-Hermite quadrature nodes and weights.
        ghq = GaussHermiteQuadrature(dim=self.D, nodes=nodes)

        # Obtain Cholevsky decomposition of the covariance matrix.
        chol_cov = chol(self.cov)

        # Adjust the Gauss-Hermite nodes using the mean of the multivariate normal.
        x_adj = np.sqrt(2) * ghq.X @ chol_cov.T + self.mu

        # Apply function to the adjusted Gauss-Hermite nodes.
        y = np.apply_along_axis(self.f, arr=x_adj, axis=1).squeeze()

        # Evaluate the integral using the corresponding weights for each adjusted node.
        self.integral_GH = 1/(pi ** (self.D/2)) * (ghq.W.T @ y)

        return self.integral_GH

    def evaluate_montecarlo(self,
                            num_sim: np.int64):

        """
            Evaluates the expectation operator with respect to the input multivariate normal distribution using
            Monte-Carlo simulations.

            :param num_sim: "Number of random draws required"
        """

        # Obtain Cholesky decomposition of the covariance matrix.
        chol_cov = chol(self.cov)

        # Random draws from standard multivariate normal adjusted to the input distribution.
        x_draws = normal(loc=0, scale=1, size=(num_sim, self.D)) @ chol_cov.T + self.mu

        # Apply function to the adjusted Gauss-Hermite nodes.
        y = np.apply_along_axis(self.f, arr=x_draws, axis=1)

        # Evaluate the integral by averaging across the function evaluated at the random draws.
        self.integral_MC = np.mean(y)

        return self.integral_MC


f = lambda x: np.sin(2 * x[0]) * x[1]**2 + x[2]

A = np.array([[1, 0.2, -0.5], [0.2, 1.6, 0.9], [-0.5, 0.9, 0.2]])

int_eval = ExpectationMultivariateNormal(func=f, dim=3,
                                         normal_dist={'Mean': np.array([0.5, -1, 0]),
                                                      'Cov': A.T @ A})
int_eval.evaluate_gauss_hermite(9)
int_eval.evaluate_montecarlo(10000000)
