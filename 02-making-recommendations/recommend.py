from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances, manhattan_distances
from scipy.stats import pearsonr
import bar_chart
import recommendations

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
        'Just My Luck': 3.0,
        'Superman Returns': 4.0,
        'The Night Listener': 4.5,
        'You, Me and Dupree': 2.5
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


def object_to_array(obj):
    labels = []
    values = []
    for _, value in enumerate(obj.keys()):
        values.append(obj[value])
        labels.append(value)
    return labels, values
labels, data = object_to_array(critics)

# Takes an array of dictionary, and normalize it
vectorizer = DictVectorizer(sparse=False)
X = vectorizer.fit_transform(data)
print("fit_transform:\n", X, "\n")
print("get_feature_names:\n", vectorizer.get_feature_names(), "\n")
print("get_params:\n", vectorizer.get_params(), "\n")


# Calculate the distance between two arrays
def euclidean(p1, p2):
    x1 = p1.reshape(1, -1)
    x2 = p2.reshape(1, -1)
    distance = pow(euclidean_distances(x1, x2)[0][0], 2)
    # The lower the distance, the higher the score
    return 1 / (1 + distance)


def similarities(labels, data, index, alg='euclidean', n=5):
    output = []
    if alg == 'euclidean':
        for i in range(len(data)):
            if i != index:
                label = labels[i]
                distance = euclidean(data[index], data[i])
                output.append((label, distance))

        # Sort tuples by the distance in descending order (highest to lowest)
        output.sort(key=lambda x: x[1], reverse=True)

        # Return only n items
        return output[:n]
    else:
        for i in range(len(data)):
            if i != index:
                label = labels[i]
                distance = pearsonr(data[index], data[i])[0]
                output.append((label, distance))

        # Sort tuples by the distance in descending order (highest to lowest)
        output.sort(key=lambda x: x[1], reverse=True)

        # Return only n items
        return output[:n]

def ranking(labels, data, index, alg='pearson'):
    sim = similarities(labels, data, index, alg, n=len(data))

    sum_sim = sum([v[1] for v in sim])
    rankings = []
    # Calculate item similarity (similarity x item)
    for k, v in enumerate(sim):
        rank =  v[1] * data[k]
        rankings.append(rank)

    # 
    si = {}
    for k, v in enumerate(rankings):
        for index, value in enumerate(v):
            si.setdefault(index, 0)
            si[index] += value
    # Similarity / sum of similiarity
    rankings = []
    for k in si:
        rankings.append((vectorizer.get_feature_names()[k], si[k] / sum_sim))
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings

# 6 = Toby 
def summary(index):
    print("\n\nTop 3 similar users to {}:\n".format(labels[index]))
    for k, v in enumerate(similarities(labels, X, index, "pearson", 3)):
        print(k + 1, v[0], v[1])

    print("\n\nRecommended items for {}:\n".format(labels[index]))
    for k, v in enumerate(ranking(labels, X, index, 6)):
        print(k + 1, v[0], v[1])

for k, v in enumerate(labels):
    summary(k)

# TODO: When recommending similar items, only recommend things that the user do not have