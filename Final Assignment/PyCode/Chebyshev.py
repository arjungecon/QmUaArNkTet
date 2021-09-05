import numpy as np
from itertools import product

# Frequently used commands
inv, ax, det = np.linalg.inv, np.newaxis, np.linalg.det
cos, pi, arccos, log = np.cos, np.pi, np.arccos, np.log


class ChebyshevApproximator:

    """
        Class used to compute Chebyshev interpolations for multidimensional functions.
    """

    def __init__(self,
                 f,
                 dim_rect: np.int32,
                 degree: np.int32,
                 nodes: np.int32,
                 bounds: np.float32):
        
        """
            :param f: Function to be approximated, function,
            :param dim_rect: The dimension of the rectangle, integer,
            :param degree: The degree of Chebyshev approximation, integer,
            :param nodes: The number of nodes in the approximation algorithm, integer,
            :param bounds: Bounds of the rectangle for the approximation, np.array of dimension dim_rect x 2
        
        """

        # Defining characteristics for the object used to perform Chebyshev approximation.
        self.f, self.dim, self.N, self.M, self.bounds = f, dim_rect, degree, nodes, bounds

        # Computation of nodes along a single dimension.
        self.nodes1D = -cos(pi * (np.arange(1, self.M + 1) - 0.5) / self.M)

        # D-fold cartesian product of Chebyshev nodes.
        self.X = np.array(list(product(self.nodes1D, repeat=self.dim)))

        # Computing the tensor product basis polynomial matrix at each D-dim Chebyshev node.
        self.T = np.apply_along_axis(lambda x: self.calculate_chebyshev_poly(val=x), 1, self.X)

        self.coeffChebyshev = 0

    def calculate_chebyshev_poly(self,
                                 val: np.float32):
        """
            Method used for degree-N Chebyshev polynomial computation at a dimension-D point.
            
            :param val: Slice at which the polynomial is evaluated, of size D
        """

        # Initial value of the recursive Chebyshev polynomial.
        chebyshev_slice = 1

        # Apply Kronecker product of computed Chebyshev polynomials over each dimension.
        for d in range(self.dim):
            chebyshev_slice = np.kron(cos(np.arange(0, self.N + 1)[ax, :] * arccos(val[d]))[ax, :], chebyshev_slice)

        return np.squeeze(chebyshev_slice)

    def calculate_chebyshev_coeff(self):
        """
            Method used to regress function evaluated at nodes on Chebyshev polynomials.
        """

        f, x, t = self.f, self.X, self.T
        a, b = self.bounds[:, 0][ax, :], self.bounds[:, 1][ax, :]

        # Converting the Chebyshev nodes to the rectangle provided in the input.
        z = 0.5 * (x + 1) * (b - a) + a

        # Evaluating the function at the adjusted Chebyshev nodes
        y = f(z)[:, ax]

        # Using the LS result to extract the Chebyshev coefficients
        self.coeffChebyshev = np.diag(np.diag(np.transpose(t) @ t) ** -1) @ np.transpose(t) @ y

    def eval_chebyshev(self,
                       values: np.float32):
        """
            Run the Chebyshev approximation routine for a given set of values.

            :param values: A vector or matrix with which to evaluate the Chebyshev approximation.
        """

        a, b = self.bounds[:, 0][ax, :], self.bounds[:, 1][ax, :]

        # Convert values from the function domain to the [-1, 1]^D rectangle.
        nu = 2 * (values - a) / (b - a) - 1

        # Apply the tensor Chebyshev basis polynomials to the transformed input values.
        t = np.apply_along_axis(lambda x: self.calculate_chebyshev_poly(val=x), 1, nu)

        # Evaluate the function using the Chebyshev coefficients associated with the Chebyshev basis polynomials.
        y_out = t @ self.coeffChebyshev

        return np.squeeze(y_out)
