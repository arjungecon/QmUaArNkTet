# Final Assignment - BUSN 37904 (Bayesian Learning)

### Setup of the Bayesian Learning model

We consider a Bayesian learning model in the spirit of Erdem and Keane (1996) or Crawford and Shum (2005). A consumer, indexed by $i$, chooses among $J$ experience goods, $j=0$ denotes the no-purchase option. The realized quality of product $j$ is
$$
\xi_{i j t} \; = \; \vartheta_{ij}+\nu_{ij t}, \qquad \nu_{ij t} \, \sim \, \mathcal{N}\left(0, \sigma_{\nu}^{2}\right)
$$
$\vartheta_{j}$ is the match value based on the product attributes and the consumer's idiosyncratic preferences over these attributes. 

Consumers have normal priors on the match values for each product,
$$
\pi_{ijt} \; \equiv \; \mathcal{N}\left(\mu_{ij t}, \sigma_{ijt}^{2}\right)
$$
 We assume that the priors are independent and that the noise terms $\nu_{j t}$ are i.i.d.

The realized utility (net of the latent utility draw $\epsilon_{j t}$ ) conditional on the purchase of product $j$ with price at time $t$ denoted by $p_{j t}$ is given by
$$
\mathfrak{u}_{j t} \; = \; \gamma-\exp \left(-\rho \xi_{j t}\right)-\alpha p_{j t}
$$
where $\rho > 0$ captures the consumer's risk aversion. $\gamma$ imposes restrictions on the market shares. Assume that the latent utility draws are Type I Extreme Value, i.i.d., and centered at 0. 

The state vector is given by $\boldsymbol{x}_{t}=\left(\boldsymbol{p}_{t}, \boldsymbol{\pi}_{t}\right)=\left(p_{1 t}, \ldots, p_{J t}, \mu_{1 t}, \sigma_{1 t}^{2}, \ldots, \mu_{J t}, \sigma_{J t}^{2}\right)$. The expected utility conditional on the purchase of $j$ given state $\boldsymbol{x}_t$ (which includes the consumer's prior belief about the match value of product $j$) is
$$
u_{j}\left(\boldsymbol{x}_{t}\right) \; = \; \mathbb{E}\left[ \mathfrak{u}_{jt} \middle| \boldsymbol{x}_{t} \right] \; = \;  \mathbb{E}\left[\gamma-\exp \left(-\rho \xi_{j t}\right)-\alpha p_{j t}\, \middle|\, \boldsymbol{x}_{t}\right]
$$
We normalize the utility from the outside option such that $u_0 \left( \boldsymbol{x}_t \right) = 0$. 

Consumers are forward-looking and discount the future using a discount factor $\beta>0 .$ Optimal choices are characterized by the choice-specific value functions $v_{j}(\boldsymbol{x})$, such that consumers choose product $j$ if and only if 
$$
v_{j}\left(\boldsymbol{x}_{t}\right)+\epsilon_{jt} \; \geq \;  v_{k}\left(\boldsymbol{x}_{t}\right)+\epsilon_{k t} \quad \forall \; k \neq j
$$
The match value for consumer $i$ from product $j$ is drawn from a normal distribution,
$$
\vartheta_{i j} \, \sim \, \mathcal{N}\left(\overline{\vartheta}_{j}, \tau_{j}^{2}\right)
$$
We assume that consumers have rational expectations such that $\pi_{i j 0} \equiv \mathcal{N}\left(\overline{\vartheta}_{j}, \tau^{2}\right)$. After a purchase (which coincides with consumption), the consumer $i$ observes the signal $\xi_{i j t}$ and updates their prior in a rational, Bayesian fashion. Given that the noisy shocks to the realized quality of the product are zero-mean, the expected value of the signal coincides with the expected value of the match value associated with attribute $j$ under the information set $\boldsymbol{x}_{t-1}$.
$$
\mathbb{E}\left[\xi_{ijt} \mid \boldsymbol{x}_{t-1} \right] \; = \; \mathbb{E}\left[\vartheta_{ij} \mid \boldsymbol{x}_{t-1} \right]
$$
