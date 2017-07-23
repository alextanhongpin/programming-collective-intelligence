import matplotlib.pyplot as plt


def draw(x, y, x_label='x_label', y_label='y_label'):
    plt.plot(x, y, 'ro')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
