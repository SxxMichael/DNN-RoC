import numpy as np


# Generate X data
def x_data(features, n, distribution=("unif", "norm", "unif1")):
    # Normal activation
    if distribution == "norm":
        x_val = np.random.normal(0, 1.0, size=(features, n))
    # Uniform activation
    elif distribution == "unif":
        x_val = np.random.uniform(0, 1.0, size=(features, n))
    elif distribution == "unif1":
        x_val = np.random.uniform(0.1, 1.0, size=(features, n))
    else:
        raise Exception('Non-supported distribution!')

    return x_val


def func(x, type=("f1", "f2", "f3", "f4")):
    n = x.shape[1]
    # return np.sin((np.pi / 2) * x) + np.exp(-x)        # uniform range (-2,2)

    # return np.power(x, 2) - np.log(np.pi/(3 * x))    # uniform range (0,4)

    # np.power(x, 2)                                   # uniform range(-2,2)
    # sample size
    # np.diagonal(np.dot(x.T, x)).reshape((1, n)) # uniform range (-2,2)

    # temp = np.exp(x[0,:] * x[1,:]) - x[2,:] * x[3,:] + np.log(x[4,:] * np.pi) - np.sqrt(5 * x[5,:]) + 7 * x[6,:] - np.pi * x[7,:] * x[8,:] + (1 / 2) * pow(x[9,:], 2)
    # return temp.reshape([1,n])        # uniform range (0.0, 1.0)

    if type == "f1":
        fun = x[0, :] + (2 * x[1, :]) + (3 * x[2, :]) + (4 * x[3, :]) + (5 * x[4, :]) 
    elif type == "f2":
        fun = np.log(x[0, :] + (2 * x[1, :]) + (3 * x[2, :]) + (4 * x[3, :]) + (5 * x[4, :]))
    elif type == "f3":
        fun = x[0, :] + (2 * x[1, :]) + (3 * x[2, :]) + (4 * x[3, :]) + (5 * x[4, :]) + (6 * x[5, :]) + (7 * x[6, :]) + (8 * x[7, :]) + (9 * x[8, :]) + (10 * x[9, :])
    elif type == "f4": 
        fun = np.exp(x[0,:] * x[1,:]) - x[2,:] * x[3,:] + np.log(x[4,:] * np.pi) - np.sqrt(5 * x[5,:]) + 7 * x[6,:] - np.pi * x[7,:] * x[8,:] + (1 / 2) * pow(x[9,:], 2)
    else:
        raise Exception('Non-supported type!')

    # return x[0, :] + (2 * x[1, :]) + (3 * x[2, :]) + (4 * x[3, :]) + (5 * x[4, :]) + (6 * x[5, :]) + (7 * x[6, :]) + (8 * x[7, :]) + (9 * x[8, :]) + (10 * x[9, :]) # uniform range (0,1)

    # return x[0, :] + (2 * x[1, :]) + (3 * x[2, :]) + (4 * x[3, :]) + (5 * x[4, :])            # uniform range (0.0, 1.0)

    # return np.log(x[0, :] + (2 * x[1, :]) + (3 * x[2, :]) + (4 * x[3, :]) + (5 * x[4, :]))    # unifrom range (0.10, 2.0)
    return fun

# Generate e data
def e_data(n, sigma):
    # Normal activation
    e_val = np.random.normal(0, sigma, n)
    return e_val


# Generate Y data
def y_data(x, e, type):
    return func(x, type) + e


def test_data(features, n, sigma, type, distribution=("unif", "norm", "unif1")):
    X = x_data(features, n, distribution)
    E = e_data(n, sigma)
    Y = y_data(X, E, type)
    true_fun = func(X, type)
    return X, Y, true_fun

# X and e include # features, # of sample
#  size 50 is min, 1000 max
