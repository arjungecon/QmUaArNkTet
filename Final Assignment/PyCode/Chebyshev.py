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
                 f: 'Function to be approximated, function',
                 dim_rect: "The dimension of the rectangle, integer",
                 degree: "The degree of Chebyshev approximation, integer",
                 nodes: 'The number of nodes in the approximation algorithm, integer',
                 bounds: 'Bounds of the rectangle for the approximation, np.array of dimension D x 2'):

        # Defining characteristics for the object used to perform Chebyshev approximation.
        self.f, self.dim, self.N, self.M, self.bounds = f, dim_rect, degree, nodes, bounds

        # Computation of nodes along a single dimension.
        self.nodes1D = -cos(pi * (np.arange(1, M + 1) - 0.5) / M)

        # D-fold cartesian product of Chebyshev nodes.
        self.X = np.array(list(product(self.nodes1D, repeat=D)))

        # Computing the tensor product basis polynomial matrix at each D-dim Chebyshev node.
        self.T = np.apply_along_axis(lambda x: self.calculateChebyshevPolynomials(val=x), 1, self.X)

        self.coeffChebyshev = 0

    def calculateChebyshevPolynomials(self,
                                      val: 'Slice at which the polynomial is evaluated, of size D'):
        """
            Method used for degree-N Chebyshev polynomial computation at a dimension-D point.
        """

        # Initial value of the recursive Chebyshev polynomial.
        cheby_slice = 1

        # Apply Kronecker product of computed Chebyshev polynomials over each dimension.
        for d in range(self.D):
            cheby_slice = np.kron(cos(np.arange(0, self.N + 1)[ax, :] * arccos(val[d]))[ax, :], cheby_slice)

        return np.squeeze(cheby_slice)

    def calculateChebyshevCoefficients(self):
        """
            Method used to regress function evaluated at nodes on Chebyshev polynomials.
        """

        f, X, T = self.f, self.X, self.T
        a, b = self.bounds[:, 0][ax, :], self.bounds[:, 1][ax, :]

        # Converting the Chebyshev nodes to the rectangle provided in the input.
        Z = 0.5 * (X + 1) * (b - a) + a

        # Evaluating the function at the adjusted Chebyshev nodes
        Y = f(Z)[:, ax]

        # Using the LS result to extract the Chebyshev coefficients
        self.coeffChebyshev = np.diag(np.diag(T.T @ T) ** -1) @ T.T @ Y

    def evaluateChebyshev(self,
                          values: 'A vector or matrix with which to evaluate the Chebyshev approximation.'):
        """
            Run the Chebyshev approximation routine for a given set of values.
        """

        a, b = self.bounds[:, 0][ax, :], self.bounds[:, 1][ax, :]

        # Convert values from the function domain to the [-1, 1]^D rectangle.
        nu = 2 * (values - a) / (b - a) - 1

        # Apply the tensor Chebyshev basis polynomials to the transformed input values.
        T = np.apply_along_axis(lambda x: self.calculateChebyshevPolynomials(val=x), 1, nu)

        # Evaluate the function using the Chebyshev coefficients associated with the Chebyshev basis polynomials.
        y_out = T @ self.coeffChebyshev

        return np.squeeze(y_out)

