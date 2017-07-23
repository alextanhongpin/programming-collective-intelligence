from sklearn.cluster import KMeans

import numpy as np
X = np.array([
    [1, 2],
    [1, 4],
    [1, 0],
    [4, 2],
    [4, 4],
    [4, 0]
])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print('kmeans:', kmeans.labels_)

r = kmeans.predict([
    [0, 0],
    [4, 4]
])
print("predict:", r)
print("cluser_centers:", kmeans.cluster_centers_)