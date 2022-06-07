Problem:

Say X and Y are independent and uniformly distributed on (0, 1). What is the expected value of X, given that X > Y?


### Knowledge point
Conditional probability

$ X ～ U(0, 1)$, and $ Y~U(0, 1) $

### Potential solution hint:

The goal is to get $$E(X| X > Y)$$.


$f(X|X > Y) $

$$ p(0 < X< x |X > Y) = \frac{p(0<X<x, X > Y)}{P(X > Y)} = \frac{\frac{1}{2}x^2}{\frac{1}{2}} = x^2
 = \int_{0}^{x} f(x|X > Y)dx $$

 $$ f(x|X > Y) = 2x $$

 Therefore, 
$$ E(x| X > Y)= \int_{0}^{1} x \cdot f(x|X > Y)dx = \int_{0}^{1} x \cdot 2xdx = \frac{2}{3} $$ 
 