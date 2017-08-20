import pandas as pd
import numpy as np
from numpy.linalg import inv
from numpy.linalg import lstsq
from sklearn.linear_model import LinearRegression

df = pd.read_csv('multiple.csv')

print(df.describe())

print(df.corr())

X = df.loc[:, ['diameter', 'toppings']]
y = df['price']

print(X.values)
print(y.values)

def reshape(arr):
    return arr.reshape(1, -1).T

X_train, X_test = X[:5].values, X[5:].values
y_train, y_test = reshape(y[:5]), reshape(y[5:])

print(X_train, X_test)
print(y_train, y_test)

# Y = XB
# B = (X.T * X)^-1 * X.T * Y

print(inv(X_train.T @ X_train) @ (X_train.T @ y_train))
print(lstsq(X_train, y_train)[0])

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
for i, prediction in enumerate(predictions):
    print('Predicted: ${:2.4f}, Target: ${}'.format(prediction[0], y_test[i][0]))

print('Score: {}'.format(model.score(X_test, y_test)))
