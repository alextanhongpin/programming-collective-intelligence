import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LassoCV

from itertools import chain

data = load_boston()

print(data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

# print(X_train, X_test)
# print(y_train, y_test)

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test.reshape(-1, 1))

# print(X_train, X_test)
# print(y_train, y_test)

y_train_flat = list(chain.from_iterable(y_train))
y_test_flat = list(chain.from_iterable(y_test))

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train_flat, cv=5)
print('Cross validation r-squared scores: {}'.format(scores))
print('Average cross validation r-squared score', np.mean(scores))

regressor.fit_transform(X_train, y_train_flat)
print('Test set r-squared score', regressor.score(X_test, y_test_flat))

# model = SelectFromModel(regressor)
# model.fit_transform(X_train, y_train_flat)
# print('Model get support', model.get_support())
# print('Model get params', model.get_params())
# for i, ok in enumerate(model.get_support()):
#     if ok:
#         print(data.feature_names[i])