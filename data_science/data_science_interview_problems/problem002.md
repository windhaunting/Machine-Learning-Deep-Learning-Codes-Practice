# Problem:

Say you have two countries of interest and want to compare variances of clicking behavior from users (i.e. total distribution of clicks). How would you do this comparison, and what assumptions need to be met?


### Knowledge points: 
 Hypothesis testing

### Potential solution hint:


We consider two countries A, B.  We get a statistics of clicks in a period of 1 month in A, and the same in B.

There are $n_1$ independent observation with a variance of clicks in A $\sigma_A$, and there are $n_2$ independent observation with a variance of clicks in B $\sigma_B$.

We assume the clicks in A and B are independent.

We use $\frac{\s_1^2}{\s_2^2$ to estimate $\frac{\sigma_A}{\sigma_B}$, 

We use hypothesis testing.
$H_0$: $\frac{\sigma_A}{\sigma_B}=1$, 
$H_1$: $\frac{\sigma_A}{\sigma_B} \neq 1$ or $\frac{\sigma_A}{\sigma_B} < 1$, or $\frac{\sigma_A}{\sigma_B} > 1$

The F-test: This test assumes the two samples come from populations that are normally distributed.
Bonett's test: this assumes only that the two samples are quantitative.
Levene's test: similar to Bonett's in that the only assumption is that the data is quantitative. Best to use if one or both samples are heavily skewed, and your two sample sizes are both under 20.

Here if we use F-test.

We use this sample:
$n_1$, $n_2$, $s_1^2$, $s_2^2$.

The test statistics F is obtained as 
$$\frac{\sigma_A}{\sigma_B} = $$.

The we get the p-value and choose a significance level (e.g. $\alpha = 0.05$ ).to get the confidence interval.

The (1-$\alpha$)100% confidence interval for $\frac{\sigma_A}{\sigma_B}$ is defined as:

$$\frac{s_1^2}{s_2^2}* F_{n_1-1, n_2-1, \alpha/2} \leq \frac{\alpha_A}{\alpha_B} \leq \frac{s_1^2}{s_2^2}* F_{n_2-1, n_1-1, \alpha/2}$$.

From the F-distribution table, we could get the value of $F_{n_1-1, n_2-1, \alpha/2}$ and $F_{n_2-1, n_1-1, \alpha/2}$

Thus, the 95% confidence interval for the ratio of the population variances is ($\frac{s_1^2}{s_2^2}* F_{n_1-1, n_2-1, \alpha/2}$, $\frac{s_1^2}{s_2^2}* F_{n_2-1, n_1-1, \alpha/2}$)

