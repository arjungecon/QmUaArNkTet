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
                 dim: np.int32,
                 nodes: np.int32):
        
        """
            :param dim: The dimension of the function, integer
            :param nodes: The number of nodes required, integer
        
        """

        # Defining characteristics for the object used to perform Gaussian-Hermite quadrature.
        self.D, self.N = dim, nodes
        self.X, self.W = np.zeros(nodes), np.zeros(nodes)

        self.derive_ghw_1d()

        if self.D > 1:

            self.derive_ghw_nd()

    def derive_ghw_1d(self):

        """
            Computes the node values and weights required.
        """

        self.X, self.W = hermgauss(deg=self.N)

    def derive_ghw_nd(self):

        """
            Computes a matrix of quadrature nodes and a vector of corresponding weights.
        """

        x_d = np.array(list(itertools.product(*(self.X,) * self.D)))
        w_d = np.prod(np.array(list(itertools.product(*(self.W,) * self.D))), 1)

        self.X, self.W = x_d, w_d
