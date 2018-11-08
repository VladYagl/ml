import numpy as np


def cal_cost(theta, X, y):
    """

    Calculates the cost for given X and Y. The following shows and example of a single dimensional X
    theta = Vector of thetas
    X     = Row of X's np.zeros((2,j))
    y     = Actual y's np.zeros((2,1))

    where:
        j is the no of features
    """

    m = len(y)

    predictions = X.dot(theta)
    cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000):
    """
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate
    iterations = no of iterations

    Returns the final theta vector and array of cost history over no of iterations
    """
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for it in range(iterations):
        prediction = np.dot(X, theta)

        theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta, X, y)

    return theta, cost_history, theta_history


test = open("../Linear/res/input.txt", "r")
features = int(test.readline())
N = int(test.readline())
inp = []
yy = []

for i in range(0, N):
    a = [float(x) for x in test.readline().split(' ')]
    inp.append(a[:-1])
    yy.append([a[-1]])

# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
X = np.array(inp)
y = np.array(yy)

scale = X.max()
print(scale)
X = X / scale
y = y / scale

lr = 0.01
n_iter = 100000
theta = np.random.randn(2, 1)

X_b = np.c_[np.ones((len(X), 1)), X]
theta, cost_history, theta_history = gradient_descent(X_b, y, theta, lr, n_iter)
theta[0] *= scale
print(X)
print(theta)
