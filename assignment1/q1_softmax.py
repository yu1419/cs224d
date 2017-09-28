import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the
    written assignment!
    """

    if len(x.shape) == 2:
        max_x = np.max(x,axis=1)
        max_x = max_x.reshape(max_x.size, 1)
        x = x - max_x
        x = np.exp(x, dtype=np.float64)
        x_sum = np.sum((x), axis=1, dtype=np.float64)
        x_sum = x_sum.reshape(x_sum.size, 1)
    else:
        max_x = np.max(x)
        x = x - max_x
        x = np.exp(x, dtype=np.float64)
        x_sum = np.sum((x), dtype=np.float64)
    x = x / x_sum
    return x

def test_softmax_basic():
    """
    Some simple tests to get you starte
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print (test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print (test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print (test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print ("You should verify these results!\n")

if __name__ == "__main__":
    test_softmax_basic()
