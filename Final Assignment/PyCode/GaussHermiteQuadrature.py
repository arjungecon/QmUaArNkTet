import numpy as np
from numpy.polynomial.hermite import hermgauss
import itertools

# Frequently used commands
inv, ax, det = np.linalg.inv, np.newaxis, np.linalg.det
cos, pi, arccos, log = np.cos, np.pi, np.arccos, np.log


class GaussHermiteQuadrature:

    """
        Class used to compute Gauss-Hermite quadrature nodes and weights in order
        to evaluate an integral.
    """

    def __init__(self,
                 Dim: "The dimension of the function, integer",
                 Nodes: "The number of nodes required, integer"):

        # Defining characteristics for the object used to perform Gaussian-Hermite quadrature.
        self.D, self.N = Dim, Nodes
        self.X, self.W = np.zeros(Nodes), np.zeros(Nodes)

        self.deriveGauHerWeights1D()

        if self.D > 1:

            self.deriveGauHerWeightsND()

    def deriveGauHerWeights1D(self):

        """
            Computes the node values and weights required.
        """

        self.X, self.W = hermgauss(deg=self.N)

    def deriveGauHerWeightsND(self):

        """
            Computes a matrix of quadrature nodes and a vector of corresponding weights.
        """

        X_d = np.array(list(itertools.product(*(self.X,) * self.D)))
        W_d = np.prod(np.array(list(itertools.product(*(self.W,) * self.D))), 1)

        self.X, self.W = X_d, W_d
