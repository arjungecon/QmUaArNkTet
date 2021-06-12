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
                 Func: "The (multivariate) function over which the expectation is computed, function",
                 Dim: "The number of inputs in the function, integer",
                 NormalDist: "Mean vector and covariance matrix of the multivariate normal distribution, dict"):

        # Defining characteristics for the object used to run the integration procedures.
        self.D, self.f = Dim, Func

        self.mu, self.cov = np.reshape(NormalDist['Mean'], (1, Dim)), NormalDist['Cov']

        self.integral_GH, self.integral_MC = None, None

    def EvaluateGaussHermite(self,
                             Nodes: "The number of nodes required to run the Gauss-Hermite quadrature, integer"):

        """
            Evaluates the expectation operator with respect to the input multivariate normal distribution using
            Gauss-Hermite quadrature.
        """

        # Initializing the Gauss-Hermite quadrature nodes and weights.
        GHQ = GaussHermiteQuadrature(Dim=self.D, Nodes=Nodes)

        # Obtain Cholesky decomposition of the covariance matrix.
        cholCov = chol(self.cov)

        # Adjust the Gauss-Hermite nodes using the mean of the multivariate normal.
        X_adj = np.sqrt(2) * GHQ.X @ cholCov.T + self.mu

        # Apply function to the adjusted Gauss-Hermite nodes.
        Y = np.apply_along_axis(self.f, arr=X_adj, axis=1)

        # Evaluate the integral using the corresponding weights for each adjusted node.
        self.integral_GH = 1/(pi ** (self.D/2)) * np.dot(Y, GHQ.W)

        return self.integral_GH

    def EvaluateMonteCarlo(self, NumSim: "Number of random draws required"):

        """
            Evaluates the expectation operator with respect to the input multivariate normal distribution using
            Monte-Carlo simulations.
        """

        # Obtain Cholesky decomposition of the covariance matrix.
        cholCov = chol(self.cov)

        # Random draws from standard multivariate normal adjusted to the input distribution.
        X_draws = normal(loc=0, scale=1, size=(NumSim, self.D)) @ cholCov.T + self.mu

        # Apply function to the adjusted Gauss-Hermite nodes.
        Y = np.apply_along_axis(self.f, arr=X_draws, axis=1)

        # Evaluate the integral by averaging across the function evaluated at the random draws.
        self.integral_MC = np.mean(Y)

        return self.integral_MC

#
# f = lambda x: np.sin(2 * x[0]) * x[1]**2 + x[2]
#
# A = np.array([[1, 0.2, -0.5], [0.2, 1.6, 0.9], [-0.5, 0.9, 0.2]])
#
# int_eval = ExpectationMultivariateNormal(Func=f, Dim=3,
#                                          NormalDist={'Mean': np.array([0.5, -1, 0]),
#                                                      'Cov': A.T @ A})
# int_eval.EvaluateGaussHermite(9)
# int_eval.EvaluateMonteCarlo(10000000)
