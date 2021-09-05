# Frequently used packages
import numpy as np
from numpy.linalg import norm
from Chebyshev import ChebyshevApproximator

# Frequently used commands
inv, ax, det = np.linalg.inv, np.newaxis, np.linalg.det
cos, pi, arccos, log = np.cos, np.pi, np.arccos, np.log

# Random number generator
unif = np.random.uniform


def func_1(X):

    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    return x * (z ** 3) + y * z + (x ** 2) * y * (z ** 2)

def func_2(X):

    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    return x * log(5 + y) * z

def func_3(X):

    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    return (x ** 2) * cos(y) * np.exp(z)


# Bounds of the rectangle over which each function is approximated.
rect_bounds = np.array([[-5, 2], [-2, 4], [-3, 3]])

# Setting up the space over which the Chebyshev approximations will be computed via vectorization.
N_vec, M_vec, func_vec = np.arange(3, 9), np.arange(7, 12), [func_1, func_2, func_3]
approx_space = np.array([(N, M, func) for func in func_vec for N in N_vec for M in M_vec if M > N + 1])

# Drawing random uniform draws from the specified rectangle.

N_draws = 10000

dim, _ = rect_bounds.shape

random_values = np.array([])

for d in range(dim):

    random_values = np.append(random_values,
                              unif(low=rect_bounds[d, 0],
                                   high=rect_bounds[d, 1],
                                   size=N_draws))


random_values = random_values.reshape([dim, N_draws]).T

def deriveChebyshevApproxError(N: 'Chebyshev polynomial degree',
                               M: 'Number of Chebyshev nodes',
                               func: 'Function to be approximated',
                               rect_bounds: 'Bounds of the rectangle for evaluation',
                               values: 'Values at which the function is approximated'):
    """
        A user-defined function that assesses the performance of the Chebyshev approximation for
        a function over a given rectangle in R^D space using a specified number of nodes and polynomial degree.
    """

    dim, _ = rect_bounds.shape

    # Initializing an object of the ChebyshevApproximator class.
    approx_obj = ChebyshevApproximator(f=func, dim_rect=dim, degree=N, nodes=M, bounds=rect_bounds)

    # Computing the Chebyshev coefficients associated with the provided function.
    approx_obj.calculate_chebyshev_coeff()

    diff = np.abs(approx_obj.eval_chebyshev(values=values) - func(random_values))

    return np.array([diff.mean(), diff.max()])


diff_res = np.apply_along_axis(
    lambda X: deriveChebyshevApproxError(
        N=X[0], M=X[1], func=X[2], rect_bounds=rect_bounds, values=random_values
        ), 1, approx_space)
