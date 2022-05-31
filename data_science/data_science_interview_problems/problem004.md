# Problem

You roll three dice and observe the sum of the three rolls. What is the probability that the sum of the outcomes is 12, given that the three rolls are different? 


### Knowledge point
Conditional probability; Bayesian theorem

### Potential solution hint:


There are total $6^3 = 216$ combinations.

Those cases are the sum of 12. There are total 22,  16 out of are different D1, D2, D3

1, 5, 6
2, 4, 6
2, 5, 5     xxxx
3, 3, 6     xxxx
3, 4, 5
3, 5, 4    
4, 2, 6
4, 3, 5
4, 4, 4     xxxxx
4, 5, 3   
4, 6, 2
5, 1, 6
5, 2, 5    xxxxx
5, 3, 4
5, 4, 3
5, 5, 2    xxxxx
5, 6, 1     
6, 1, 5
6, 2, 4
6, 3, 3   xxxxx
6, 4, 2 
6, 5, 1 



$$ P(S=12|D1, D2, D3 \; different) = \frac{P(D1, D2, D3 \; different|S=12) P(S=12)}{P(D1, D2, D3 \; different)} = \frac{16/22 * 22/(6^3)}{6!/(6^3)} = 13.33% $$