
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from itertools import chain

# Training data
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]


def drawPlot(X, y, X_predict, y_predict):
    plt.figure()
    plt.title('Pizza price plotted against diameter')
    plt.xlabel('Diameter in inches')
    plt.ylabel('Prices in dollars')
    original_legend, = plt.plot(X, y, 'k.', label='originals')
    predicted_legend, = plt.plot(X_predict, y_predict, 'r+', label='predicted')
    plt.legend(handles=[original_legend, predicted_legend])
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    plt.show()

# Create and fit the model
model = LinearRegression()

model.fit(X, y)
print('A 12" pizza should cost: ${}'.format(model.predict([[12]])[0][0]))

# Make a prediction for a range of pizza's diameter
X_predict, y_predict = [], []
for i in range(25):
    pred = model.predict([[i]])[0]
    X_predict.append(i)
    y_predict.append(pred)
    # print('Predicted price ${} for diameter {}'.format(pred[0], i))

# Plot the outcome for comparison
# drawPlot(X, y, X_predict, y_predict)


# Evaluating the fitness of a model with cost function
rss = np.mean((model.predict(X) - y) ** 2)
print('Residual sum of squares: {}'.format(rss))

variance = np.var(X, ddof=1)
print('Variance: {}'.format(variance))


X_flat = list(chain.from_iterable(X))
y_flat = list(chain.from_iterable(y))
covariance = np.cov(X_flat, y_flat)[0][1]
print('Covariance: {}'.format(covariance))

beta = covariance / variance
print('Beta: {}'.format(beta))

# Given y = a + Bx
# To solve a = y - Bx
alpha = np.mean(y) - beta * np.mean(X)
print('Alpha: {}'.format(alpha))
print('Linear model: y = a + bX = {} + {}x'.format(alpha, beta))

def linear_model (x):
    return alpha + beta * x

for i in range(25):
    print('When diameter is {} then the observed price is ${} and the predicted price is ${}'.format(i, linear_model(i), y_predict[i][0]))


# The real data will be compared with the test data above
# TODO: Rename X and y above to X_test and y_test respectively
X_test = [[8], [9], [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]]

print('R-squared: {}'.format(model.score(X_test, y_test)))

