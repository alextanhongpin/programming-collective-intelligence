import matplotlib.pyplot as plt

import numpy as np
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

boston = load_boston()
X, y = boston['data'], boston['target']

# We use the base estimator LassoCV since the L1 norm promotes
# sparsity of features. clf = classifier
clf = LassoCV()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

print(n_features)

# Reset the threshold until the number of features equals two.
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

plt.title('Features selected from Boston using SelectFromModel with threshold {:0.3f}'.format(sfm.threshold))

feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]

plt.plot(feature1, feature2, 'r.')
plt.xlabel('Feature number 1')
plt.ylabel('Feature number 2')
plt.ylim([np.min(feature2), np.max(feature1)])
plt.show()