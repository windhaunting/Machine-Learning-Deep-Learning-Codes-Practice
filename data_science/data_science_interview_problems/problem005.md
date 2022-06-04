# Problem

 Suppose you roll a fair die n times, where n is very large. What is the average time between occurrences of a given number?


### Knowledge point
Probability, Geometric Distribution

### Potential solution hint:

If I understand correctly, the question is to ask how many times of rolls needed on average for the same given number appearing again. It should be the same answer for any number between 1 and 6. 
e.g. if the given number is 3, how many rolls needed on average to get 3 again.


For each face from 1 to 6 in a fair dire, each has a 1/6 prob appearing in one roll.
$ p = \frac{1}{6} $

Then it means the given number appears second again at kth times of roll before after (k-1)th of times try.
The prob of this is that
 $$ p(Y = k) = \binom{k-1}{1}p^1(1-p)^{k-2}p^{1} = (k-1)(1-p)^{k-2}p^{2}
 $$

Then we got the expected number 

$$ E(Y) = \sum_{k=2}^{n}k \cdot P(Y=k)=\sum_{k=2}^{n} k \cdot (k-1)(1-p)^{k-2}p^{2} = \sum_{k=2}^{n} k(k-1)(1-p)^{k-2}p^{2} $$

Other question reference:

https://www.quora.com/If-you-roll-a-die-600-times-about-how-many-times-would-you-expect-to-roll-a-4

https://www.cis.jhu.edu/~xye/papers_and_ppts/ppts/SolutionsToFourProblemsOfRollingADie.pdf