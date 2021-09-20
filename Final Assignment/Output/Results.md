# Final Assignment - BUSN 37904 (Bayesian Learning)

## Model Ingredients

### Bayesian Learning Component

We consider a Bayesian learning model in the spirit of Erdem and Keane (1996) or Crawford and Shum (2005). A consumer (typically indexed by $i$ but we ignore this notation) chooses among $J$ experience goods, $j=0$ denotes the no-purchase option. The realized quality of product $j$ is
$$
\xi_{j t} \; = \; \vartheta_{j}+\nu_{j t}, \qquad \nu_{j t} \, \sim \, \mathcal{N}\left(0, \sigma_{\nu}^{2}\right)
$$
$\vartheta_{j}$ is the match value based on the product attributes and the consumer's idiosyncratic preferences over these attributes. 

Consumers have normal priors on the match values for each product, where at time $t$, the consumer believes that $\vartheta_j$ is distributed as $\pi_{jt}$ where
$$
\pi_{jt} \; \equiv \; \mathcal{N}\left(\mu_{j t}, \sigma_{jt}^{2}\right)
$$
We assume that the priors are independent of each other and that the noise terms $\nu_{j t}$ are i.i.d. draws across $j$ and $t$. The match value for each consumer from product $j$ is drawn from a normal distribution,
$$
\vartheta_{j} \, \sim \, \mathcal{N}\left(\overline{\vartheta}_{j}, \tau_{j}^{2}\right)
$$
We assume that consumers have rational expectations such that $\pi_{j 0} \equiv \mathcal{N}\left(\overline{\vartheta}_{j}, \tau^{2}\right)$ wherein consumers begin by assuming the mean of their matching value is the average matching value. 
$$
\mathbb{E}\left[ \xi_{jt} \mid \boldsymbol{x}_{0} \right] \; = \; \overline{\vartheta}_j
$$
After a purchase (which coincides with consumption), the consumer $i$ observes the signal $\xi_{i j t}$ and updates their prior in a rational, Bayesian fashion. Given that the noisy shocks to the realized quality of the product are zero-mean, the expected value of the signal coincides with the expected value of the match value associated with attribute $j$ under the information set $\boldsymbol{x}_{t-1}$.
$$
\mathbb{E}\left[\xi_{jt} \mid \boldsymbol{x}_{t-1} \right] \; = \; \mathbb{E}\left[\vartheta_{j} \mid \boldsymbol{x}_{t-1} \right] \; = \; \mu_{j,t-1}
$$
Let $C_{jt} = 1$ if the consumer chooses product $j$ and $0$ otherwise. Using standard Bayesian updating results given normality assumptions, the mean of the prior is updated from $t-1$ to $t$ as shown below: 
$$
\mu_{jt} \; := \;   \mu_{j,t-1}+ C_{jt} \lambda_{j,t-1} \left( \xi_{jt} - \mu_{j,t-1}  \right) \\
\text{with Kalman gain} \;\; 
\lambda_{jt} \; = \; \frac{\sigma^2_{j,t-1}}{\sigma^2_{j,t-1} + \sigma^2_{\nu}}
$$
 Furthermore, the variance of the prior distribution is updated as follows:
$$
\sigma^2_{j,t} \; = \; {\left(\frac{1}{\sigma^2_{j,t-1}} + \frac{C_{j,t}}{\sigma^2_{\nu}} \right)}^{-1}
$$
Note that the priors for the remaining $J - 1$ products remain unchanged when good $j$ is chosen by the consumer.

### Characterizing the Bellman Equation

The realized utility (net of the latent utility draw $\epsilon_{j t}$) conditional on the purchase of product $j$ with price at time $t$ denoted by $p_{j t}$ is given by
$$
\mathfrak{u}_{j t} \; = \; \gamma-\exp \left(-\rho \xi_{j t}\right)-\alpha p_{j t}
$$
where $\rho > 0$ captures the consumer's risk aversion. $\gamma$ imposes restrictions on the market shares. Assume that the latent utility draws are Type I Extreme Value, i.i.d., and centered at 0. 

The state vector is given by $\boldsymbol{x}_{t}=\left(\boldsymbol{p}_{t}, \boldsymbol{\pi}_{t}\right)=\left(p_{1 t}, \ldots, p_{J t}, \mu_{1 t}, \sigma_{1 t}^{2}, \ldots, \mu_{J t}, \sigma_{J t}^{2}\right)$. The expected utility conditional on the purchase of $j$ given state $\boldsymbol{x}_t$ (which includes the consumer's prior belief about the match value of product $j$) is
$$
\begin{align*}
u_{j}\left(\boldsymbol{x}_{t}\right) \; = \; \mathbb{E}\left[ \mathfrak{u}_{jt} \middle| \boldsymbol{x}_{t} \right] \; & = \;  \mathbb{E}\left[\gamma-\exp \left(-\rho \xi_{j t}\right)-\alpha p_{j t}\, \middle|\, \boldsymbol{x}_{t}\right]\\
& = \; \gamma - \exp \left( -\rho \mu_{jt} + \frac{\rho^2}{2} \sigma^2_{jt} \right) - \alpha p_{jt} 
\end{align*}
$$
We normalize the utility from the outside option such that $u_0 \left( \boldsymbol{x}_t \right) = 0$. Furthermore, the price vectors $\mathbf{p}_{t}$ are i.i.d. draws from a discrete distribution with a support of $\left\{\mathfrak{p}_{1}, \ldots, \mathfrak{p}_{K}\right\}$ with probability mass function given by $\omega_{k}=\operatorname{Pr}\left\{\boldsymbol{p}_{t}=\boldsymbol{\mathfrak{p}}_{k}\right\}$.

Consumers are forward-looking and discount the future using a discount factor $\beta>0 .$ Optimal choices are characterized by the choice-specific value functions $v_{j}(\boldsymbol{x})$, such that consumers choose product $j$ at time $t$ ($C_{jt} = 1$) if and only if 
$$
v_{j}\left(\boldsymbol{x}_{t}\right)+\epsilon_{jt} \; \geq \;  v_{k}\left(\boldsymbol{x}_{t}\right)+\epsilon_{k t} \quad \forall \; k \neq j
$$
The Bellman equation facing a consumer at time $t$ with state variable $\boldsymbol{x}_t$ is given by
$$
v\left(\boldsymbol{x}_{t}, \boldsymbol{\varepsilon}_{t}\right) \; = \; \max _{j}  \Big\{ \mathbb{E}_{{\boldsymbol{\pi}_{t}}}\left[\gamma-\exp \left(\rho \xi_{j t}\right)-\alpha p_{j t}+\varepsilon_{j t} \right]+ \beta \, \mathbb{E}_{\boldsymbol{x}_{t+1}, \varepsilon_{t+1}} \left[ v \left(\boldsymbol{x}_{t+1}, \boldsymbol{\varepsilon}_{t+1}\right)  \right] \Big\}
$$
We can take expectations with respect to both $\boldsymbol{p}_t$ and $\boldsymbol{\varepsilon}_t$ in the above Bellman equation as they are both i.i.d. across $t$, and arrive at choice-specific value functions $v_j$ in the expression on the RHS and the expected value function given beliefs $\boldsymbol{\pi}_t$ on the LHS.
$$
\begin{align*}
\overline{v}(\boldsymbol{\pi}_t) \; &= \; \sum_{k=1}^K \omega_k \, \int \max_j  \Big\{ \mathbb{E}_{{\boldsymbol{\pi}_{t}}}\left[\gamma-\exp \left(\rho \xi_{j t}\right)-\alpha p_{kj}+\varepsilon_{j t} \right]+ \beta \, \mathbb{E}_{\boldsymbol{x}_{t+1}, \varepsilon_{t+1}} \left[ v \left(\boldsymbol{x}_{t+1}, \boldsymbol{\varepsilon}_{t+1}\right)  \right] \Big\} \; g(\boldsymbol\varepsilon) \,\operatorname{d}\boldsymbol\varepsilon \\
& = \;  \sum_{k=1}^K \omega_k \, \int \max_j  \left\{ v_j(\mathbf{\boldsymbol{\mathfrak{p}}_k, \boldsymbol{\pi}_t}) +  \varepsilon_{j t} \right\} \; g(\boldsymbol\varepsilon) \,\operatorname{d}\boldsymbol\varepsilon
\end{align*}
$$
We assume that the price vector is i.i.d. across time periods as it reduces the state space of our model. If we do not assume prices are i.i.d., the state space gets very large and the problem becomes quite computationally difficult. For instance, we will need to include lagged prices as a state variable as the probabilities $\{\omega_{k,t}\}$ would now be a function of $\boldsymbol{p}_{t-1}$. This assumption is economically sound as we anticipate the consumer to fully account for prices in the utility function.

Given the idiosyncratic shocks are i.i.d. Gumbel distributed, we can simplify the expression for the expected value function in terms of the choice-specific value functions.
$$
\begin{align*}
\overline{v}(\boldsymbol{\pi}_t)  
\; &= \; \sum_{k=1}^K \omega_k \, \log \left(\sum_j \exp   v_j(\mathbf{\boldsymbol{\mathfrak{p}}_k, \boldsymbol{\pi}_t})   \right)
\end{align*}
$$

### Numerical Approximations

Note that the choice-specific value function is now given by
$$
v_j(\mathbf{\boldsymbol{\mathfrak{p}}_k, \boldsymbol{\pi}_t}) \; = \; u_j(\mathbf{\boldsymbol{\mathfrak{p}}_k, \boldsymbol{\pi}_t}) + \int_{\boldsymbol{\pi}}\overline{v}(\boldsymbol{\pi}_{t+1}) \, \text{f}\left( \boldsymbol{\pi}_{t+1} \mid \boldsymbol{\pi}_{t}, j\right) \operatorname{d}\boldsymbol{\pi}_{t+1}
$$
where $\text{f}\left( \boldsymbol{\pi}_{t+1} \mid \boldsymbol{\pi}_{t}, j\right)$ denotes the state transition probability when the consumer chooses product $j$ and updates their beliefs about the match value. We can interpolate the expected value function using Chebyshev polynomials. Assuming we use $K$ polynomials and that $\mathbf{T}(\boldsymbol{\pi})$ denotes the $2J$-tensor product of Chebyshev polynomials, we have:
$$
\overline{v}(\boldsymbol{\pi}) \; \equiv \; \sum_{k = 1}^K \theta_k \, T_k(\boldsymbol{\pi})
$$
Furthermore, since $\text{f}\left( \boldsymbol{\pi}_{t+1} \mid \boldsymbol{\pi}_{t}, j\right)$ corresponds to the p.d.f. of a multivariate normal distribution, we can use the Gauss-Hermite quadrature to evaluate the integral.

<!--However, the consumer only chooses one product in each time period and the beliefs corresponding to the match value of that product are updated. This implies that only one pair of $\mu_{j,t+1}$ and $\sigma_{j,t+1}$ are updated at any given time, and therefore the expectation needs to be evaluated-->

This implies that the integral used to characterize the choice-specific value function can be rewritten as shown, assuming that we use $Q$ quadrature nodes.
$$
\begin{align*}
\int_{\boldsymbol{\pi}}\overline{v}(\boldsymbol{\pi}_{t+1}) \, \text{f}\left( \boldsymbol{\pi}_{t+1} \mid \boldsymbol{\pi}_{t}, j\right) \operatorname{d}\boldsymbol{\pi}_{t+1} \; & \approx \; \int_{\boldsymbol{\pi}} \sum_{k = 1}^K \theta_k \, T_k(\boldsymbol{\pi_{t+1}})\, \text{f}\left( \boldsymbol{\pi}_{t+1} \mid \boldsymbol{\pi}_{t}, j\right) \operatorname{d}\boldsymbol{\pi}_{t+1} \\
& = \; \sum_{k = 1}^K \theta_k \int_{\boldsymbol{\pi}} T_k(\boldsymbol{\pi_{t+1}})\, \text{f}\left( \boldsymbol{\pi}_{t+1} \mid \boldsymbol{\pi}_{t}, j\right) \operatorname{d}\boldsymbol{\pi}_{t+1}  \\
& \equiv \; \sum_{k = 1}^K \theta_k \,  E_k (\boldsymbol{\pi_t}, j) \\
& \approx \;  \sum_{k = 1}^K \theta_k \sum_{q = 1}^Q  \psi_{q}^{(j)} \, T_k (\boldsymbol{\pi}_{t+1, q})
\end{align*}
$$
where we see that we now evaluate the expectation over the Chebyshev polynomials using Gauss-Hermite quadrature. 

### Updating the Value Function

The numerical approximations outlined above will expedite the value function iteration. At iteration $n$, the Chebyshev approximation coefficients $\boldsymbol{\theta}^{(n)}$ represent the optimal approximation of the expected value function $\overline{v}$. We can then evaluate each choice-specific value function as shown:
$$
\begin{align*}
	v^{(n)}_j(\mathbf{\boldsymbol{\mathfrak{p}}_k, \boldsymbol{\pi}_t}) \; \approx \; u_j(\mathbf{\boldsymbol{\mathfrak{p}}_k, \boldsymbol{\pi}_t}) + \beta \sum_{k = 1}^K \theta_k^{(n)} \sum_{q = 1}^Q  \psi_{q}^{(j)} \, T_k (\boldsymbol{\pi}_{t+1, q} \, ; \, j)
\end{align*}
$$
These can be used to update the expected value function in the subsequent iteration
$$
\overline{v}^{(n+1)}(\mathbf{x}) \; \longleftarrow \; \sum_{k=1}^K \omega_k \, \log \left(\sum_j \exp   v_j^{(n)}(\mathbf{\boldsymbol{\mathfrak{p}}_k, \boldsymbol{\pi}})   \right)
$$
We obtain the updated Chebyshev coefficient vector $\theta^{(n+1)}$ based on the regression
$$
\boldsymbol{\theta}^{(n+1)} \; = \; \left(\boldsymbol{T}(\pi)' \,\boldsymbol{T}(\pi)\right)^{-1} \boldsymbol{T}(\pi)'\, \overline{v}^{(n+1)}(\mathbf{x})
$$
The updated value function $v^{(n+1)}$ is then given by $v^{(n+1)}(\boldsymbol{x})=\sum_{k} \theta_{k}^{(n+1)} T_{k}(\boldsymbol{x})$.





​	





