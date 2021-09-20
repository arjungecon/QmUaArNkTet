import timeit

import numpy as np
from BayesianLearning import ValueFunction
from ChebyshevApproximator import ChebyshevApproximator
from GaussHermiteQuadrature import GaussHermiteQuadrature
from ExpectationMultivariateNormal import ExpectationMultivariateNormal

initializer = {
    'product_count': 2,
    'price_support_size': 2,
    'prices': np.array([[1.2, 1.2], [3, 3.5]]).T,
    'price_probability_vector': np.array([0.16, 0.84]),
    'util_market_share': 3,
    'util_CARA': 1,
    'util_elasticity': 1.5,
    'discount_factor': 0.998,
    'match_quality_mean': np.array([10., 20.]),
    'match_quality_var': np.array([[5., 0.], [0., 10.]]),
    'prior_var': 10. * np.eye(2),
    'match_shock_var': 2,
    'Chebyshev_nodes': 7,
    'Chebyshev_degrees': 3,
    'GHQuad_nodes': 7,
    'VFI_max_iteration': 1e3,
    'VFI_tolerance': 2e-1,
    'VFI_verbose': True,
    'VFI_verbose_frequency': 5,
    'VFI_relaxation': 1
}

obj = ValueFunction(initializer=initializer)

A, B = obj.get_evf(prior_mean=obj.prior_mean, prior_var=obj.prior_var)

