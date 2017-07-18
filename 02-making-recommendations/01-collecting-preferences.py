from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances, manhattan_distances
from scipy.stats import pearsonr

critics = {
    'Lisa Rose': {
        'Lady in the Water': 2.5,
        'Snakes on a Plane': 3.5,
        'Just My Luck': 3.0,
        'Superman Returns': 3.5,
        'You, Me and Dupree': 2.5,
        'The Night Listener': 3.0
    },
    'Gene Seymour': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 3.5,
        'Just My Luck': 1.5,
        'Superman Returns': 5.0,
        'The Night Listener': 3.0,
        'You, Me and Dupree': 3.5
    },
    'Michael Phillips': {
        'Lady in the Water': 2.5,
        'Snakes on a Plane': 3.0,
        'Superman Returns': 3.5,
        'The Night Listener': 4.0
    },
    'Claudia Puig': {
        'Snakes on a Plane': 3.5,
        'Just My Luck': 2.0,
        'Superman Returns': 3.0,
        'The Night Listener': 3.0,
        'You, Me and Dupree': 2.0
    },
    'Mick LaSalle': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 4.0,
        'Just My Luck': 2.0,
        'Superman Returns': 3.0,
        'The Night Listener': 3.0,
        'You, Me and Dupree': 2.0
    },
    'Jack Matthews': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 4.0,
        'The Night Listener': 3.0,
        'Superman Returns': 5.0,
        'You, Me and Dupree': 3.5
    },
    'Toby': {
        'Snakes on a Plane': 4.5,
        'Superman Returns': 4.0,
        'You, Me and Dupree': 1.0
    }
}

data = []
for _, name in enumerate(critics.keys()):
    data.append(critics[name])

vectorizer = DictVectorizer(sparse=False)
X = vectorizer.fit_transform(data)
print(X)
print(vectorizer.get_feature_names())
print(vectorizer.get_params())


def euclidean(p1, p2):
    return 1 / (1 + pow(euclidean_distances(p1, p2)[0][0],2))

print(X[3], X[5])
print(euclidean(X[3], X[5]))
# print 1 / (1 + pairwise_distances(X[3], X[5]))
# print 1 / (1 + manhattan_distances(X[3], X[5]))
print(pearsonr(X[3], X[5])[0])

def similarities(alg='euclidean', n=5):
    output = []
    if alg == 'euclidean':
        for i in range(len(X)):
            for j in range(len(X)):
                if i != j:
                    output.append(euclidean(X[i], X[j]))
        output = list(set(output))
        output.sort()
        output.reverse()
        return output[:n]
    else:
        for i in range(len(X)):
            for j in range(len(X)):
                if i != j:
                    output.append(pearsonr(X[i], X[j])[0])
        output = list(set(output))
        output.sort()
        output.reverse()
        return output[:n]

print(similarities(alg='pearson', n=5))

