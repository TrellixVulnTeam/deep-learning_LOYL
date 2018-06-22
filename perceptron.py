# Python 3

import numpy as np

n = 3

x1 = np.array([[-1], [1]])
x2 = np.array([[0], [-1]])
x3 = np.array([[10], [1]])
xs = [x1, x2, x3]

y1 = 1
y2 = -1
y3 = 1
ys = [y1, y2, y3]

def classified_correctly(w, b, i):
    """
    returns 1 if the ith example is classified correctly given w and b,
    otherwise returns 0.
    """
    if ys[i] * (w.T @ xs[i] + b) > 0:
        return 1
    else:
        return 0

def test_classified_correctly():
    """
    throws error if tests for classified_correctly don't pass.
    """
    w_0 = np.array([[0], [0]])
    b_0 = 0
    for i in range(n):
        assert(not classified_correctly(w_0, b_0, i))

def next_example(w, b, start_i):
    """
    returns the index of the next incorrectly classified example using weights w
    and bias b beggining at start_i. If all are correct, returns -1.
    """
    for i in list(range(start_i, n)) + list(range(0, start_i)):
        if not classified_correctly(w, b, i):
            return i
    return -1

def perceptron_iteration(w, b, i):
    """
    returns the resulting weight and bias after convergence of the perceptron
    learning algorithm, where the algorithm starts at example i with weight
    w and bias b.
    """
    i = next_example(w, b, i)
    if i == -1:
        return w, b
    x = xs[i]
    y = ys[i]
    w_new = w + (y * x)
    b_new = b + y
    return perceptron_iteration(w_new, b_new, i)

def perceptron_learning():
    """
    prints the result of running the perceptron learning algorithm with inital
    weight vector [[0], [0]] and bias of 0.
    """
    w_0 = np.array([[0], [0]])
    b_0 = 0
    w, b = perceptron_iteration(w_0, b_0, 0)
    print("Resulting weight: ", w)
    print("Resulting bias: ", b)

perceptron_learning()


