import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
df = pd.read_csv('./linear.csv', delimiter=',', encoding="utf-8")

# List the index
print(df.index)

print(df.corr())

# List the columns name
print(df.columns)

# Print a summary of your data
print(df.describe())

# Print the first three rows
print(df.head(3))


model = LinearRegression()
X = df['diameter'].values
y = df['price'].values

def reshape(arr):
    return arr.reshape(1, -1).T

X_train, X_test = reshape(X[:5]), reshape(X[5:])
y_train, y_test = reshape(y[:5]), reshape(y[5:])



model.fit(X_train, y_train)
model.score(X_test, y_test)


output = model.predict([[12]])[0][0]
print('A 12" pizza will cost ${:2.2f}'.format(output))

# Sorting by values
# print(df.sort_values(by='diameter'))