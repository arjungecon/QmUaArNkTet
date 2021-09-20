import numpy as np
import itertools
from scipy.special import logsumexp as logsumexp
from numpy.linalg import norm as norm
from numpy.linalg import cholesky as chol
import pandas as pd

from ChebyshevApproximator import ChebyshevApproximator
from GaussHermiteQuadrature import GaussHermiteQuadrature
from ExpectationMultivariateNormal import ExpectationMultivariateNormal
from ParallelHelper import parallel_apply_along_axis as par_aaa

# Frequently used commands
inv, ax, det = np.linalg.inv, np.newaxis, np.linalg.det
cos, pi, arccos, log = np.cos, np.pi, np.arccos, np.log
sin, arcsin, exp = np.sin, np.arcsin, np.exp


class ValueFunction:

    def __init__(self, initializer,
                 ):

        """
            :param initializer: Dictionary that initializes dimensions
                within the learning model.

            J is the number of products. (product_count)
            L is the number of possible price vector realizations.
                (price_support_size)

            𝝼 is the noise in the realized match quality.
        """

        self.J = initializer['product_count']
        self.L = initializer['price_support_size']

        # Matrix of dimension J x L
        self.price = initializer['prices']

        # Vector of size L
        self.price_prob = initializer['price_probability_vector']

        # Utility function parameters.
        self.gamma = initializer['util_market_share']
        self.rho = initializer['util_CARA']
        self.alpha = initializer['util_elasticity']

        # Discounting.
        self.beta = initializer['discount_factor']

        # Value Function Iteration Fixed Point Settings.
        self.max_ite = initializer['VFI_max_iteration']
        self.tol = initializer['VFI_tolerance']
        self.verb = initializer['VFI_verbose']
        self.verb_freq = initializer['VFI_verbose_frequency']
        self.relax = initializer['VFI_relaxation']

        # Match quality: Multivariate normal distribution
        self.mq_mean = initializer['match_quality_mean']
        self.mq_var = initializer['match_quality_var']
        
        # Prior distribution
        self.prior_mean = self.mq_mean
        self.prior_var = initializer['prior_var']

        # Noisy signal distribution
        self.nu = initializer['match_shock_var']  # Mean of shock is zero.

        # Gauss-Hermite quadrature properties + object
        self.K = initializer['GHQuad_nodes']
        self.GHQ = GaussHermiteQuadrature(dim=self.J, nodes=self.K)

        # Chebyshev Approximation properties + object
        # Note that the adjusted GHQ nodes give rise to the bounds for the Chebyshev object.
        self.M = initializer['Chebyshev_nodes']
        self.N = initializer['Chebyshev_degrees']

        # Utility Matrix: Create utility corresponding to J products and L prices.
        # Add another layer at the top for the outside option set at 0.
        self.U_mat = self.utility(prior_mean=self.mq_mean,
                                  prior_var=self.prior_var,
                                  price_vectors=self.price)
        self.U_mat = np.r_[np.ones([1, self.L]), self.U_mat]
        # matrix(J + 1 (product), L (price vectors))

    def expectation_chebyshev(self, mean, var):

        """
            Computes the expectation of the Chebyshev polynomials with respect to the underlying input
            multivariate normal distribution using the Gauss-Hermite quadrature.

            :param mean: Mean of the multivariate normal distribution, typically of size self.J.
            :param var: Variance of the multivariate normal distribution, typically of dimension self.J.
        """

        ghq_adj = np.sqrt(2) * self.GHQ.X @ chol(var).T + mean

        chebyshev_approx = ChebyshevApproximator(
            dim_rect=self.J,
            degree=self.N,
            nodes=self.M,
            bounds=np.array(
                [
                    [np.min(ghq_adj[:, i]), np.max(ghq_adj[:, i])] for i in range(ghq_adj.shape[1])
                ]
            )
        )

        exp_chebyshev_pol = ExpectationMultivariateNormal(
            func=lambda x: chebyshev_approx.calculate_chebyshev_poly_multiple(val=x, inside_bounds=False),
            dim=self.J,
            normal_dist={'Mean': mean, 'Cov': var}
        ).evaluate_gauss_hermite(nodes=self.K)

        return exp_chebyshev_pol

    def utility(self, prior_mean, prior_var, price_vectors):

        """
            Computes the expected utility value (or matrix) for given match quality prior beliefs and
                set of price vector(s) for the products.

            :param prior_mean: Vector of size self.J containing mean of product-specific prior beliefs.
            :param prior_var: Vector of size self.J containing variance of product-specific prior beliefs.
            :param price_vectors: Matrix of self.J x self.L containing vectors of prices associated
                with the products.

            :return: Matrix of self.J x self.L containing utilities associated with each set of match qualities
                and prices for each product.
        """

        return self.gamma - exp(
            - self.rho * prior_mean[:, ax] + (self.rho ** 2)/2 * np.diag(prior_var + self.nu)[:, ax]
        ) - self.alpha * price_vectors

    def bayesian_updating(self, prior_mean, prior_var, choice, signal):

        """
            Computes the posterior mean vector corresponding to a signal associated with a product.

            :param prior_mean: Vector of size self.J containing posterior mean product-specific match qualities.
            :param prior_var: Matrix of self.J x self.J containing posterior covariance.
            :param choice: Product chosen, within range(self.J) + 1.
            :param signal: Value of match quality signal corresponding to the chosen product.

            :return: Mean of posterior distribution.
        """

        i = choice - 1  # Reindex to Python format

        # Compute Kalman gain associated with the product.
        kalman_gain = prior_var[i, i] / (prior_var[i, i] + self.nu)

        # Computed updated posterior mean and variance corresponding to each product.
        updated_mean = prior_mean[i] + kalman_gain * (signal - prior_mean[i])
        updated_var = (1 / prior_var[i, i] + 1 / self.nu) ** -1

        post_mean, post_var = prior_mean.copy(), prior_var.copy()
        post_mean[i], post_var[i, i] = updated_mean, updated_var

        return [post_mean, post_var]

    def get_choice_specific_vf(self, prior_mean, prior_var, evf_theta):

        """
            Computes the choice-specific value functions for all products across all price realizations.

            :param prior_mean: Vector of size self.J containing prior mean product-specific match qualities.
            :param prior_var: Matrix of size self.J x self.J containing prior covariance of product-specific
                match qualities.
            :param evf_theta: Chebyshev coefficients of size (self.N + 1) ** self.J that represent the expected
                value function.
        """

        # Generate set of possible signals from the GHQ applied to the consumer's prior.
        ghq_nodes_signal = np.sqrt(2) * self.GHQ.X @ chol(
            prior_var[:] + self.nu * np.eye(self.J)).T + prior_mean[:]

        # Construct the expected utility using the prior mean and variance fed in. Set outside option to 0.
        utility = self.utility(prior_mean=prior_mean[:], prior_var=prior_var[:],
                               price_vectors=self.price)  # J x K matrix
        utility = np.r_[np.ones([1, self.L]), utility]

        def compute_continuation_value(choice):

            """
                Computes the continuation value term inside the CSVF for a specific choice including the
                    outside option net of the discount factor.
            """

            def vec_beliefs(x):

                """
                    Applies the Bayesian updating function to a vector of signals for a single product.
                """
                return self.bayesian_updating(prior_mean=prior_mean.copy(),
                                              prior_var=prior_var.copy(),
                                              signal=x, choice=choice)[0]

            if choice == 0:    # Outside option

                # The expected Chebyshev polynomial representation depends only on the prior mean
                # and variance seen by the consumer in the case of the outside option.
                exp_t_mat = self.expectation_chebyshev(mean=prior_mean[:], var=prior_var[:])

            else:   # A product chosen

                i = choice - 1

                # Computes the updated posterior variance of the match quality corresponding to
                # the product chosen.
                post_var = prior_var.copy()
                post_var[i, i] = (1 / prior_var[i, i] + 1 / self.nu) ** -1

                # Computes the vector of posterior means associated with each signal realization
                # corresponding to the product.
                post_mean = np.vectorize(vec_beliefs, signature='()->(n)')(ghq_nodes_signal[:, i])

                # Computes the Chebyshev polynomial vector associated with each set of posterior beliefs
                # corresponding to the generated signal.
                t_mat = np.apply_along_axis(
                    lambda x: self.expectation_chebyshev(mean=x, var=post_var[:]), 1, post_mean
                )

                # Computes the expected Chebyshev polynomial vector over all the possible posterior
                # beliefs using the GHQ weights.
                exp_t_mat = 1 / (pi ** (self.J / 2)) * (self.GHQ.W.T @ t_mat)

            # Computes the continuation term in the CSVF using the Chebyshev coefficients that represent
            # the Expected Value Function.
            return exp_t_mat @ evf_theta

        evf = self.beta * np.vectorize(compute_continuation_value)(np.arange(self.J + 1))

        return utility + evf[:, ax]

    def get_evf(self, prior_mean, prior_var):

        """
            Solves for the Expected Value Function as a fixed point solution to the
                Bellman equation.

            :param prior_mean: Vector of size self.J containing prior mean product-specific match qualities.
            :param prior_var: Matrix of size self.J x self.J containing prior covariance of product-specific
                match qualities.
        """
        
        list_ite = list()   # Create list that stores each iteration's details.

        ite_count = 0   # Iteration count tracker
        error = 1e4     # Measures the norm between the previous theta and the updated theta

        # Create a Chebyshev approximator object with bounds set 2 deviations around the prior mean.
        chebyshev_approx = ChebyshevApproximator(
            dim_rect=self.J,
            degree=self.N,
            nodes=self.M,
            bounds=np.array(
                [[(prior_mean[i] - 2 * np.sqrt(prior_var[i, i])),
                  (prior_mean[i] + 2 * np.sqrt(prior_var[i, i]))] for i in range(self.J)]
            ))

        # Chebyshev coefficients used to represent the EVF; a guess.
        ite_theta = chebyshev_approx.coeffChebyshev.squeeze()
        ite_theta[0] = 550

        # Chebyshev nodes generated from the object inside the specified bounds.
        chebyshev_nodes = chebyshev_approx.X_adj

        # Running the Value Function Iteration algorithm.
        while ite_count <= self.max_ite and error > self.tol:

            # Compute the choice-specific value functions for all products and prices
            # with each Chebyshev node used as the prior mean vector for the customer's beliefs.
            csvf = par_aaa(
                func1d=lambda x: self.get_choice_specific_vf(
                    prior_mean=x, prior_var=prior_var, evf_theta=ite_theta),
                axis=1, arr=chebyshev_nodes
            )
            # Matrix of size M^J x J + 1 x L

            # Calculating the EVF using the multinomial logit result
            evf = np.einsum('j, ij -> i', self.price_prob, logsumexp(csvf, axis=1))

            # Computing the new Chebyshev coefficients to represent the EVF.
            theta_new = (chebyshev_approx.pre_mat @ evf[:, ax]).squeeze()[:]
            theta_old = ite_theta[:]

            # Distance between the previous and new guess of the EVF coefficients.
            error = norm(theta_new - theta_old, ord=1)

            ite_theta = self.relax * theta_new[:] + (1 - self.relax) * ite_theta[:]

            if self.verb and (ite_count % self.verb_freq == 0):
                print('[{}] error = {:2.4e} theta = {}'.format(
                    ite_count, error, np.abs(theta_new - theta_old)))

            ite_count += 1

            list_ite.append([ite_count - 1, evf, theta_new])

        return ite_theta, list_ite

