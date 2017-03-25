import optimization
import oracles
import sklearn.datasets as ds
import os

import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nose.tools import assert_almost_equal, ok_, eq_

# import plot_trajectory_2d

import numpy as np
import numpy.linalg as li
import scipy.sparse as sp

np.random.seed(7412)

# ------------ Checking hess vec
A = np.random.rand(4, 4)
b = np.random.rand(4)
v = np.random.rand(4)
x = np.random.rand(4)
tr = oracles.create_log_reg_oracle(A, b, 1)

print(tr.hess_vec(x, v))
print(np.array(tr.hess(x)).dot(v))

for i in range(5):
    x = np.random.rand(4)
    ok_(np.allclose(tr.hess_vec(x, v), oracles.hess_vec_finite_diff(tr.func, x, v), rtol=1e-1, atol=1e-1))



# print(tr.hess(x).dot(tr.grad(x)))
# print(tr.grad(x))
# print(tr.hess_vec(x, tr.grad(x)))
# ------------ Experiment 1

for d in ['exp1', 'exp2', 'exp3', 'exp4']:
    if not os.path.exists(d):
        os.makedirs(d)

plt.clf()

for lnn, c in zip(range(1, 5), ['green', 'red', 'blue', 'black']):
    n = 10**lnn
    for i in range(15):
        Aa = lambda k: sp.diags(np.concatenate((np.random.uniform(1, k, n-2), np.array([1, k])), axis=0), 0)
        b = np.random.uniform(1, 20, n)
        def T(k):
            # quad = oracles.QuadraticOracle(Aa(k), b)
            Amat = Aa(k)
            x_star, msg, history = optimization.conjugate_gradients(lambda x: Amat.dot(x), b,
                                                                    np.random.uniform(10, 30, n), trace=True)
            # x_star, msg, history = optimization.gradient_descent(quad, np.random.uniform(10, 30, n), trace=True)
            return len(history['time'])
        grid = np.array(list(range(3, 60)))*2
        results = np.vectorize(T)(grid)
        plt.plot(grid, results, color=c, label="Dim = {}".format(n))
plt.xlabel('k')
plt.ylabel('T(n, k)')
plt.savefig("exp1/Tkn.png")

plt.clf()
for lnn, c in zip(range(1, 5), ['green', 'red', 'blue', 'black']):
    n = 10 ** lnn
    res = []
    for i in range(15):
        Aa = lambda k: sp.diags(np.concatenate((np.random.uniform(1, k, n - 2), np.array([1, k])), axis=0), 0)
        b = np.random.uniform(1, 20, n)
        def T(k):
            # quad = oracles.QuadraticOracle(Aa(k), b)
            Amat = Aa(k)
            x_star, msg, history = optimization.conjugate_gradients(lambda x: Amat.dot(x), b,
                                                                    np.random.uniform(10, 30, n), trace=True)
            # x_star, msg, history = optimization.gradient_descent(quad, np.random.uniform(10, 30, n), trace=True)
            return len(history['time'])
        grid = np.array(list(range(3, 120)))
        res.append(np.vectorize(T)(grid))
    plt.plot(grid, np.mean(res, axis=0), color=c, label="Dim = {}".format(n))
plt.xlabel('k')
plt.ylabel('T(n, k)')
plt.savefig("exp1/Tkn-mean.png")

# ----------------- Experiment 2

if not os.path.exists('datasets'):
    print('\033[91m' + "Create directory 'datasets' near this script and move all datasets there" + '\033[0m')
    exit(0)

A, b = ds.load_svmlight_file("datasets/{}".format("gisette_scale"))
lamda = 1 / np.alen(b)
hists = dict()

logr = oracles.create_log_reg_oracle(A, b, lamda, oracle_type='optimized')

ls = [0, 1, 5, 10, 50, 100]
colors = ["blue", "green", "red", "yellow", "orange", "black"]

for l in ls:
    print("Computing {}".format(l))
    x_star, msg, history = optimization.lbfgs(logr, np.zeros(A.shape[1]), trace=True, memory_size=l)
    hists[l] = history

df_0 = li.norm(logr.grad(np.zeros(A.shape[1])))**2

plt.clf()
for l, c in zip(ls[1:], colors[1:]):
    r_k = np.vectorize(lambda x: math.log(li.norm(x**2)/df_0))(hists[l]['grad_norm'])
    plt.plot(range(len(hists[l]['time'])), r_k, color = c)
plt.xlabel('Iteration')
plt.ylabel('r_k')
plt.savefig('exp2/r_k-positive-l-iter.png')

plt.clf()
for l, c in zip(ls[1:], colors[1:]):
    r_k = np.vectorize(lambda x: math.log(li.norm(x**2)/df_0))(hists[l]['grad_norm'])
    plt.plot(hists[l]['time'], r_k, color = c)
plt.xlabel('Seconds')
plt.ylabel('r_k)')
plt.savefig('exp2/r_k-positive-l-seconds.png')

plt.clf()
for l, c in zip(ls, colors):
    r_k = np.vectorize(lambda x: math.log(li.norm(x**2)/df_0))(hists[l]['grad_norm'])
    plt.plot(range(len(hists[l]['time'])), r_k, color = c)
plt.xlabel('Iteration')
plt.ylabel('r_k')
plt.savefig('exp2/r_k-iter.png')

plt.clf()
for l, c in zip(ls, colors):
    r_k = np.vectorize(lambda x: math.log(li.norm(x**2)/df_0))(hists[l]['grad_norm'])
    plt.plot(hists[l]['time'], r_k, color = c)
plt.xlabel('Seconds')
plt.ylabel('r_k)')
plt.savefig('exp2/r_k-seconds.png')

# ------------------ Experiment 3

for data in ["w8a", "rcv1_train.binary", "gisette_scale", "real-sim"]:
    A, b = ds.load_svmlight_file("datasets/{}".format(data))
    lamda = 1 / np.alen(b)
    hists = dict()
    logr = oracles.create_log_reg_oracle(A, b, lamda, oracle_type='optimized')
    for opt, label, c in zip([optimization.hessian_free_newton, optimization.gradient_descent, optimization.lbfgs],
                             ["HFN", "GD", "L-BFGS"],
                             ["green", "blue", "red"]):
        print("Computing {} - {}".format(data, label))
        x_star, msg, history = opt(logr, np.zeros(A.shape[1]), trace=True)
        hists[label] = history
    plt.clf()
    plt.plot(hists["GD"]['time'], hists["GD"]['func'], color='green')
    plt.plot(hists["HFN"]['time'], hists["HFN"]['func'], color='blue')
    plt.plot(hists["L-BFGS"]['time'], hists["L-BFGS"]['func'], color='red')
    plt.xlabel('Seconds')
    plt.ylabel('F(x)')
    plt.savefig('exp3/func-sec-{0}.png'.format(data))

    plt.clf()
    plt.plot(range(len(hists["GD"]['time'])), hists["GD"]['func'], color='green')
    plt.plot(range(len(hists["HFN"]['time'])), hists["HFN"]['func'], color='blue')
    plt.plot(range(len(hists["L-BFGS"]['time'])), hists["L-BFGS"]['func'], color='red')
    plt.xlabel('Iterations')
    plt.ylabel('F(x)')
    plt.savefig('exp3/func-iter-{0}.png'.format(data))

    plt.clf()
    df_0 = li.norm(logr.grad(np.zeros(A.shape[1])))**2
    r_k_gd = np.vectorize(lambda x: math.log(li.norm(x**2)/df_0))(hists['GD']['grad_norm'])
    r_k_new = np.vectorize(lambda x: math.log(li.norm(x**2)/df_0))(hists['HFN']['grad_norm'])
    r_k_bf = np.vectorize(lambda x: math.log(li.norm(x ** 2) / df_0))(hists['L-BFGS']['grad_norm'])
    plt.plot(hists["GD"]['time'], r_k_gd, color='green')
    plt.plot(hists["HFN"]['time'], r_k_new, color='blue')
    plt.plot(hists["L-BFGS"]['time'], r_k_bf, color='red')
    plt.xlabel('Seconds')
    plt.ylabel('ln(r_k)')
    plt.savefig('exp3/r_k-{0}.png'.format(data))



# ---------------------- Expetiment 4


for i in [0, 1, 10, 50]:
    _A = np.random.rand(20, 20)
    A = _A.T.dot(_A)
    b = np.random.rand(20)
    oracle = oracles.QuadraticOracle(A, b)
    x0 = np.zeros(20)
    x1, msg1, hist1 = optimization.conjugate_gradients(lambda v: oracle.hess_vec(x0, v), b, x0, trace=True)
    x2, msg2, hist2 = optimization.lbfgs(oracle, x0, trace=True, line_search_options={'method': 'Best'}, memory_size=i)
    plt.clf()
    plt.plot(range(len(hist1['time'])), np.vectorize(lambda s: math.log(s))(hist1['residual_norm']), 'r-')
    plt.plot(range(len(hist2['time'])), np.vectorize(lambda s: math.log(s))(hist2['grad_norm']), 'b--')
    plt.xlabel('Iteration')
    plt.ylabel('log(r_k)')
    plt.savefig('exp4/r_k-{}'.format(i))
