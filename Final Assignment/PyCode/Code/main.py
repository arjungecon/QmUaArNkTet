import numpy as np
from BayesianLearning import BayesianLearning
from datetime import datetime
import os
import pickle
import bz2
import _pickle as cpickle
from ChebyshevApproximator import ChebyshevApproximator
from GaussHermiteQuadrature import GaussHermiteQuadrature
from ExpectationMultivariateNormal import ExpectationMultivariateNormal

initializer = {
    'product_count': 2,
    'price_support_size': 2,
    'prices': np.array([[1, 1.1], [1, 1.2]]).T,
    'price_probability_vector': np.array([0.4, 0.6]),
    'util_market_share': 1,
    'util_CARA': 1,
    'util_elasticity': 1.5,
    'discount_factor': 0.998,
    'match_quality_mean': np.array([7., 7.]),
    'match_quality_var': np.array([[4., 0.], [0., 6.]]),
    'prior_var': 3. * np.eye(2),
    'match_shock_var': 2,
    'Chebyshev_nodes': 5,
    'Chebyshev_degrees': 3,
    'GHQuad_nodes': 7,
    'VFI_max_iteration': 1000,
    'VFI_tolerance': 0.1,
    'VFI_verbose': True,
    'VFI_verbose_frequency': 1,
    'VFI_relaxation': 1,
    'Learning_horizon': 100,
    'Simulation_seed': 12345
}

obj = BayesianLearning(initializer=initializer)

mq_draw = obj.rng.multivariate_normal(mean=obj.mq_mean, cov=obj.mq_var)
price_trajectory = obj.rng.choice(a=obj.L, size=obj.max_time, p=obj.price_prob)

res_dict = obj.bayesian_learning(match_quality=mq_draw,
                                 price_trajectory=price_trajectory)
res_dict['mq_draw'] = mq_draw
res_dict['initializer'] = initializer

os.chdir('/home/arjung/quantmarketing/final/Pickles')
pkl_name = 'res' + datetime.today().strftime("%d-%m-%Y-%H-%M")
with bz2.BZ2File(pkl_name + '.pbz2', 'w') as f:
    cpickle.dump(res_dict, f)
os.chdir('..')
