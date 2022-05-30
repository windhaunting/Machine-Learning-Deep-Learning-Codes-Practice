# Problem

We have N observations for some variable which we model as being drawn from a Gaussian distribution. What are your best guesses for the parameters of the distribution? Derive it mathematically.


### Knowledge point

Maximum Likelihood Estimation


### Potential solution hint:

There are N observations from a population assuming being drawn from a Gaussian distribution. The population distribution is assumed to be normal distribution $\mu_0$ and $\sigma_0$, that has the probability density function for a given value $x_i$.

$$f(x_i) = \frac{1}{\sigma_{0}\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x_i-\mu_{0}}{\sigma_{0}})^2}$$

$\mu_0$ and $\sigma_0$ are the parameters to be estimated for our data so that we can match our data to its most likely Gaussian bell curve, that is, looking for a curve that maximizes the probability of our data given a set of curve parameters


**Two assumptions** needed: 

* Data must be independently distributed.

* Data must be identically distributed.


Ref: https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
