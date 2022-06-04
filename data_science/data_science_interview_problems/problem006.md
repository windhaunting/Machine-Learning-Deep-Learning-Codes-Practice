# Problem

Say that the lifetime of electric vehicles are modeled using a Gaussian distribution. Each type of electric vehicle has an expected lifetime and a lifetime variance. Say you chose two different types of electric vehicles at random. What is the probability that the two lifetimes will be within n time units?

### Knowledge point
Probability, Normal Distribution


### Potential solution hint:

Assume the lifetime of different types of electric vehicles (EV) are independent normal distribution

Assume the first type of EV lifetime is X with mean $\mu_1$ and $\sigma_1$ the second type of EV lifetime is Y with mean $\mu_2$ and $\sigma_2$.

Then the problem is to get the prob of $P(|X-Y| <= n)$

Generally, if $X_i$ ($i = 1,2,..,n$) are independent normal random variables with means $\mu_i$ and variances $\sigma_i$, and $c_i$ ($i = 1,2,..,n$) are constants, $Y=c_1X_1 + c_2X_2 +...+ c_nX_n$ is normal with mean $\mu_Y=c_1\mu_1 +...+ c_n\mu_n$ and variance $c_1^2\sigma_1^2+...+c_n^2\sigma_n^2$.

Therefore $X-Y$ follows the normal distribution with the mean $\mu_1 - \mu_2$ and variance $\sigma_1^2 + \sigma_2^2$



