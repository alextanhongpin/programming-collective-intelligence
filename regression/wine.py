import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split, cross_val_score

df = pd.read_csv('winequality-red.csv', sep=';')

print(df.describe())

def plot_graph():
    plt.scatter(df['alcohol'], df['quality'])
    plt.title('Alcohol against quality')
    plt.xlabel('Alcohol')
    plt.ylabel('Quality')
    plt.show()

print(df.corr())

X = df[list(df.columns[:-1])]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)
print('R-squared: {}'.format(regressor.score(X_test, y_test)))
scores = cross_val_score(regressor, X, y, cv=5)
print(scores.mean(), scores)

plt.scatter(y_predictions, y_test)
plt.title('Predicted quality against true quality')
plt.xlabel('True quality')
plt.ylabel('Predicted quality')
plt.show()