import optimization
import oracles
import sklearn.datasets as ds
import os

import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_trajectory_2d

from nose.tools import assert_almost_equal, ok_, eq_

import numpy as np
import numpy.linalg as li
import scipy.sparse as sp

# ----------------------- Checking grad and hess

A = np.random.rand(10, 4)
b = np.random.rand(10)
tr = oracles.create_log_reg_oracle(A, b, 1)
for i in range(5):
    x = np.random.rand(4)
    ok_(np.allclose(tr.grad(x), oracles.grad_finite_diff(tr.func, x), rtol=1e-1, atol=1e-1))

# ---------------------- Experiment 1

for d in ['exp1', 'exp2', 'exp3', 'exp4']:
    if not os.path.exists(d):
        os.makedirs(d)

exps = [
    {
        "A": np.array([[5, 1], [1, 1]]),
        "b": np.zeros(2),
        "st": "normal"
    },
    {
        "A": np.array([[25, 2], [2, 1]]),
        "b": np.array([0, 0]),
        "st": "stratched"
    }
]

for ix, x_0 in enumerate([np.array([3, 4]), np.array([1, 25])]):
    for ip, params in enumerate(exps):
        for met in ["Wolfe", "Armijo", "Constant"]:
            plt.clf()
            qua1 = oracles.QuadraticOracle(params['A'], params['b'])
            plot_trajectory_2d.plot_levels(qua1.func)
            x_star, msg, history = optimization.gradient_descent(qua1, x_0, trace=True, line_search_options={'method': met, 'c': 0.1})
            plot_trajectory_2d.plot_trajectory(qua1.func, history['x'])
            plt.savefig("exp1/{0}-{1}-{2}.png".format(x_0, params['st'], met))

for e in exps:
    print("{0} - {1}".format(e['st'], li.cond(e['A'])))

# ------------------------ Experiment 2

np.random.seed(7412)

plt.clf()

for lnn, c in zip(range(1, 5), ['green', 'red', 'blue', 'black']):
    n = 10**lnn
    for i in range(15):
        Aa = lambda k: sp.diags(np.concatenate((np.random.uniform(1, k, n-2), np.array([1, k])), axis=0), 0)
        b = np.random.uniform(1, 20, n)
        def T(k):
            quad = oracles.QuadraticOracle(Aa(k), b)
            x_star, msg, history = optimization.gradient_descent(quad, np.random.uniform(10, 30, n), trace=True)
            return len(history['grad_norm'])
        grid = np.array(list(range(3, 60)))*2
        results = np.vectorize(T)(grid)
        plt.plot(grid, results, color=c, label="Dim = {}".format(n))
plt.xlabel('k')
plt.ylabel('T(n, k)')
plt.savefig("exp2/Tkn.png")

plt.clf()
for lnn, c in zip(range(1, 5), ['green', 'red', 'blue', 'black']):
    n = 10 ** lnn
    res = []
    for i in range(15):
        Aa = lambda k: sp.diags(np.concatenate((np.random.uniform(1, k, n - 2), np.array([1, k])), axis=0), 0)
        b = np.random.uniform(1, 20, n)
        def T(k):
            quad = oracles.QuadraticOracle(Aa(k), b)
            x_star, msg, history = optimization.gradient_descent(quad, np.random.uniform(10, 30, n), trace=True)
            return len(history['grad_norm'])
        grid = np.array(list(range(3, 120)))
        res.append(np.vectorize(T)(grid))
    plt.plot(grid, np.mean(res, axis=0), color=c, label="Dim = {}".format(n))
plt.xlabel('k')
plt.ylabel('T(n, k)')
plt.savefig("exp2/Tkn-mean.png")

# ----------------------- Experiment 4

np.random.seed(31415)
m, n = 10000, 8000
A = np.random.randn(m, n)
b = np.sign(np.random.randn(m))
lamda = 1 / np.alen(b)
hists = dict()

for opt in ["usual", "optimized"]:
    logr = oracles.create_log_reg_oracle(A, b, lamda, opt)
    x_star, msg, history = optimization.gradient_descent(logr, np.zeros(A.shape[1]), trace=True)
    hists[opt] = history
plt.clf()
plt.plot(range(len(hists["usual"]['time'])), hists["usual"]['func'], color='green')
plt.plot(range(len(hists["optimized"]['time'])), hists["optimized"]['func'], color='blue')
plt.xlabel('Iteration')
plt.ylabel('F(x)')
plt.savefig('exp4/func-iter.png')
plt.clf()
plt.plot(hists["usual"]['time'], hists["usual"]['func'], color='green')
plt.plot(hists["optimized"]['time'], hists["optimized"]['func'], color='blue')
plt.xlabel('Seconds')
plt.ylabel('F(x)')
plt.savefig('exp4/func-second.png')
plt.clf()
df_0 = li.norm(logr.grad(np.zeros(A.shape[1])))**2
r_k_gd = np.vectorize(lambda x: math.log(li.norm(x**2)/df_0))(hists['usual']['grad_norm'])
r_k_new = np.vectorize(lambda x: math.log(li.norm(x**2)/df_0))(hists['optimized']['grad_norm'])
plt.plot(hists["usual"]['time'], r_k_gd, color='green')
plt.plot(hists["optimized"]['time'], r_k_new, color='blue')
plt.xlabel('Seconds')
plt.ylabel('ln(r_k)')
plt.savefig('exp4/r_k.png')

# ----------------------- Experiment 3

np.random.seed(7412)

if not os.path.exists('datasets'):
    print('\033[91m' + "Create directory 'datasets' near this script and move all datasets there" + '\033[0m')
    exit(0)

for data in ["w8a", "gisette_scale"]:
    A, b = ds.load_svmlight_file("datasets/{}".format(data))
    lamda = 1 / np.alen(b)
    hists = dict()
    logr = oracles.create_log_reg_oracle(A, b, lamda)
    for opt, label, c in zip([optimization.gradient_descent, optimization.newton], ["GD", "Newton"], ["green", "blue"]):
        x_star, msg, history = opt(logr, np.zeros(A.shape[1]), trace=True)
        hists[label] = history
    plt.clf()
    plt.plot(hists["GD"]['time'], hists["GD"]['func'], color='green')
    plt.plot(hists["Newton"]['time'], hists["Newton"]['func'], color='blue')
    plt.xlabel('Seconds')
    plt.ylabel('F(x)')
    plt.savefig('exp3/func-{0}.png'.format(data))
    plt.clf()
    df_0 = li.norm(logr.grad(np.zeros(A.shape[1])))**2
    r_k_gd = np.vectorize(lambda x: math.log(li.norm(x**2)/df_0))(hists['GD']['grad_norm'])
    r_k_new = np.vectorize(lambda x: math.log(li.norm(x**2)/df_0))(hists['Newton']['grad_norm'])
    plt.plot(hists["GD"]['time'], r_k_gd, color='green')
    plt.plot(hists["Newton"]['time'], r_k_new, color='blue')
    plt.xlabel('Seconds')
    plt.ylabel('ln(r_k)')
    plt.savefig('exp3/r_k-{0}.png'.format(data))

A, b = ds.load_svmlight_file("datasets/real-sim")

lamda = 1 / np.alen(b)
hists = dict()
logr = oracles.create_log_reg_oracle(A, b, lamda)
for opt, label, c in zip([optimization.gradient_descent], ["GD"], ["green"]):
    x_star, msg, history = opt(logr, np.zeros(A.shape[1]), trace=True)
    hists[label] = history
plt.clf()
plt.plot(hists["GD"]['time'], hists["GD"]['func'], color='green')
plt.xlabel('Seconds')
plt.ylabel('F(x)')
plt.savefig('exp3/func-{0}.png'.format("sample-real-sim"))
plt.clf()
df_0 = li.norm(logr.grad(np.zeros(A.shape[1]))) ** 2
r_k_gd = np.vectorize(lambda x: math.log(li.norm(x ** 2) / df_0))(hists['GD']['grad_norm'])
plt.plot(hists["GD"]['time'], r_k_gd, color='green')
plt.xlabel('Seconds')
plt.ylabel('ln(r_k)')
plt.savefig('exp3/r_k-{0}.png'.format("sample-real-sim"))
