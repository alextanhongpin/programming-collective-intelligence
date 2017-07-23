from math import sqrt

def pearson(prefs, p1, p2):
    si={}

    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1
    
    n = len(si)

    # If they have no ratings in common, return 0
    if n == 0:
        return 0

    sum1 = sum(prefs[p1][it] for it in si)
    sum2 = sum(prefs[p2][it] for it in si)

    sum_squares1 = sum([pow(prefs[p1][it], 2) for it in si])
    sum_squares2 = sum([pow(prefs[p2][it], 2) for it in si])

    sum_products = sum([prefs[p1][it] * prefs[p2][it] for it in si])

    num = sum_products - (sum1 * sum2 / n)
    den = sqrt((sum_squares1 - pow(sum1, 2) / n) * (sum_squares2 - pow(sum2, 2) / n))
    if den == 0:
        return 0
    
    r = num / den
    return r

def euclidean(prefs, person1, person2):
    # Similarities
    si={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
    
    # If they have no ratings in common, return 0
    if len(si) == 0:
        return 0
    
    # Add up the squares of all differences
    sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item], 2) 
                      for item in prefs[person1] 
                      if item in prefs[person2]])
    return 1 / (1 + sum_of_squares)
