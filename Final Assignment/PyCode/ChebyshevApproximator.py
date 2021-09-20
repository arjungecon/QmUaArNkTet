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
                 dim_rect: np.int32,
                 degree: np.int32,
                 nodes: np.int32,
                 bounds):
        
        """
            :param dim_rect: The dimension of the rectangle, integer,
            :param degree: The degree of Chebyshev approximation, integer,
            :param nodes: The number of nodes in the approximation algorithm, integer,
            :param bounds: Bounds of the rectangle for the approximation, np.array of dimension dim_rect x 2
        
        """

        # Defining characteristics for the object used to perform Chebyshev approximation.
        self.dim, self.N, self.M, self.bounds = dim_rect, degree, nodes, bounds

        # Computation of nodes along a single dimension.
        self.nodes1D = -cos(pi * (np.arange(1, self.M + 1) - 0.5) / self.M)

        # D-fold cartesian product of Chebyshev nodes.
        self.X = np.array(list(product(self.nodes1D, repeat=self.dim)))

        a, b = self.bounds[:, 0][ax, :], self.bounds[:, 1][ax, :]

        # Converting the Chebyshev nodes to the rectangle provided in the input.
        self.X_adj = self.shift_outside_bounds(val=self.X)

        # Computing the tensor product basis polynomial matrix at each D-dim Chebyshev node.
        self.T = np.apply_along_axis(lambda x: self.calculate_chebyshev_poly(val=x), 1, self.X)

        # Computing the regression-pre object for the coefficients.
        self.pre_mat = np.diag(np.diag(np.transpose(self.T) @ self.T) ** -1) @ np.transpose(self.T)

        self.coeffChebyshev = np.zeros((self.T.shape[1], 1))

    def calculate_chebyshev_poly(self,
                                 val: np.float32):
        """
            Method used for degree-N Chebyshev polynomial computation at a single dimension-D point inside
            the [-1, 1]^D box.
            
            :param val: Slice at which the polynomial is evaluated, of size D
        """

        # Initial value of the recursive Chebyshev polynomial.
        chebyshev_slice = 1

        # Apply Kronecker product of computed Chebyshev polynomials over each dimension.
        for d in range(self.dim):
            chebyshev_slice = np.kron(cos(np.arange(0, self.N + 1)[ax, :] * arccos(val[d]))[ax, :], chebyshev_slice)

        return np.squeeze(chebyshev_slice)

    def shift_inside_bounds(self, val):

        """
            Method used to shift multiple dimension-D points from the bounds to the box [-1, 1]^D.

            :param val: Array of size J x D where J is the number of dimension-D points.
        """

        a, b = self.bounds[:, 0][ax, :], self.bounds[:, 1][ax, :]

        val = 2 * (val - a) / (b - a) - 1

        return val

    def shift_outside_bounds(self, val):

        """
            Method used to shift multiple dimension-D points from [-1, 1]^D to the bounds specified.

            :param val: Array of size J x D where J is the number of dimension-D points.
        """

        a, b = self.bounds[:, 0][ax, :], self.bounds[:, 1][ax, :]

        val = 0.5 * (val + 1) * (b - a) + a

        return val

    def calculate_chebyshev_poly_multiple(self, val,
                                          inside_bounds: bool = False):

        """
            Method used for degree-N Chebyshev polynomial computation at multiple dimension-D points.

            :param val: Array of size J x D where J is the number of dimension-D points at which we evaluate
             the Chebyshev polynomials.
            :param inside_bounds: Set to True if the points are adjusted to be inside [-1, 1]^D.

             Returns a matrix of size J x (N + 1)^D where N is the degree chosen.
        """

        if not inside_bounds:

            val = self.shift_inside_bounds(val=val)

        return np.apply_along_axis(lambda x: self.calculate_chebyshev_poly(val=x), arr=val, axis=1)

    def calculate_chebyshev_coeff(self, func):
        """
            Method used to regress function evaluated at nodes on Chebyshev polynomials.

            :param func: Function to be approximated, function,
        """

        # Evaluating the function at the adjusted Chebyshev nodes
        y = func(self.X_adj)[:, ax]

        # Using the LS result to extract the Chebyshev coefficients
        self.coeffChebyshev = self.pre_mat @ y

        return self.coeffChebyshev

    def eval_chebyshev(self,
                       values: np.float32):
        """
            Run the Chebyshev approximation routine for a given set of values.

            :param values: A vector or matrix with which to evaluate the Chebyshev approximation.
        """

        a, b = self.bounds[:, 0][ax, :], self.bounds[:, 1][ax, :]

        nu = 2 * (values - a)/(b - a) - 1

        # Apply the tensor Chebyshev basis polynomials to the transformed input values.
        t = np.apply_along_axis(lambda x: self.calculate_chebyshev_poly(val=x), 1, nu)

        # Evaluate the function using the Chebyshev coefficients associated with the Chebyshev basis polynomials.
        y_out = t @ self.coeffChebyshev

        return np.squeeze(y_out)
