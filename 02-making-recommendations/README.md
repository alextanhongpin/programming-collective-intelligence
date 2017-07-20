# Recommendations Algorithm

## Content-Based

If Person A likes an Item A, Person A might like items similar to Item A.


## Collaborative Filtering

If Person A likes item 1, 2 and 3 and Person B likes 2, 3 and 4, and Person A is similar to Person B, then Person A should like item 4 and Person B should like item 1.

Can be further separated into three parts:

- User-User collaborative filtering
- Item-Item collaborative filtering
- Market basket analysis


## Example

Euclidean Distance between Lisa Rose and Gene Seymour:

| Rating \ User | Lisa Rose `(x)` | Gene Seymour `(y)` | Distance `(x - y)` | Distance Squared `(x - y)^2` |
|--|--|--|--|--|
| Lady in the Water | 2.5 | 3.0 | -0.5 | 0.25 | 
| Snakes on a Plane | 3.5 | 3.5 | 0.0 | 0.0 |
| Just My Luck' | 3.0 | 1.5 | 1.5 | 2.25 | 
| Superman Returns' | 3.5 | 5.0 | -1.5 | 2.25 | 
| You, Me and Dupree' | 2.5 | 3.5 | -1.0 | 1.0 |
| The Night Listener' | 3.0 | 3.0 | 0.0 | 0.0 |
| Sum of Squares `(r)` | | | | 5.75 |
| Euclidean Distance `(1 / (1 + r)`) | | | | 0.1481 |

Pearson Distance between Lisa Rose and Gene Seymour:


| Rating \ User | Lisa Rose `(x)` | Gene Seymour `(y)` | Product Distance `(x * y)` | `(x^2)` | `(y^2)` 
|--|--|--|--|--|--|
| Lady in the Water | 2.5 | 3.0 | 7.50 | 6.25 | 9.00 | 
| Snakes on a Plane | 3.5 | 3.5 | 12.25 | 12.25 | 12.25 | 
| Just My Luck' | 3.0 | 1.5 | 4.50 | 9.00 | 2.25 | 
| Superman Returns' | 3.5 | 5.0 | 17.50 | 12.25 | 25.00 |
| You, Me and Dupree' | 2.5 | 3.5 | 8.75 | 6.25 | 12.25 |
| The Night Listener' | 3.0 | 3.0 | 9.0 | 9.00 | 9.00 |
| Sum | `sum1`=18.0 | `sum2`=19.5 | `pSum`=59.5 | `sum1sq`=55.00 | `sum2sq`=69.75 |

```
Numerator = pSum - (sum1 * sum2 / n) 
 = 59.5 - (18.0 * 19.5 / 6)
 = 1.0
 
Denominator = sqrt((sum1sq - (pow(sum1,2)/n)) * (sum2sq - (pow(sum2,2)/n)))
 = 2.524
 
 Pearson Distance = Numerator / Denominator
 = 1 / 2.524
 = 0.3960
```
