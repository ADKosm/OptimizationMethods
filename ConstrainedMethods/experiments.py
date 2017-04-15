import optimization
import oracles
import sklearn.datasets as ds
import os

import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nose.tools import assert_almost_equal, ok_, eq_

import numpy as np
import numpy.linalg as li
import scipy.sparse as sp

np.random.seed(7412)

# -------------- Experiment 1 ----------

for d in ['exp1', 'exp2', 'exp3', 'exp4']:
    if not os.path.exists(d):
        os.makedirs(d)

plt.clf()

for x_0, label in zip([np.zeros(5), np.array([3, 5, 8, 2, 5]), np.array([3, 5, 8, 2, 5])*10, np.array([3, 5, 8, 2, 5])*100], ["Zero", "Near", "Far", "ReallyFar"]):
    plt.clf()
    res = []
    for a_0 in [a / 10 for a in range(1, 50)]:
        A = np.random.rand(5, 5)
        b = np.random.rand(5)
        r = 1/5
        oracle = oracles.create_lasso_nonsmooth_oracle(A, b, r)
        x_star, msg, hist = optimization.subgradient_method(oracle, x_0, alpha_0=a_0, trace=True, max_iter=10**4)
        res.append(len(hist['func']))
    plt.plot([a / 10 for a in range(1, 50)], res)
    plt.xlabel("Alpha")
    plt.ylabel("Iterations")
    plt.savefig("exp1/{}.png".format(label))

# ------------------ Experiment 2

for n in [5, 50]:
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    r = 1/n

    oracle = oracles.create_lasso_prox_oracle(A, b, r)
    x_star, msg, hist = optimization.proximal_gradient_descent(oracle, np.zeros(n), trace=True)
    plt.clf()
    plt.plot(range(len(hist['line_counter'])), hist['line_counter'])
    plt.xlabel("Iteration")
    plt.ylabel("Line search counter")
    plt.savefig("exp2/LCounter{}.png".format(n))

# ------------------- Experiment 3

for n in [50, 1000]:
    for eps, eps_name in zip([10**-2, 10**-7], ["low", "high"]):
        A = np.random.rand(n, n)
        b = np.random.rand(n)
        r = 1
        x_0 = np.zeros(n)
        u_0 = np.ones(n) * 50
        hists = dict()
        for method, orac, lab in zip([optimization.subgradient_method, optimization.proximal_gradient_descent, optimization.barrier_method_lasso],
                                     [oracles.create_lasso_nonsmooth_oracle, oracles.create_lasso_prox_oracle, None],
                                     ["sub", "prox", "bar"]):
            if lab == "bar":
                x_star, msg, hist = method(A, b, r, x_0, u_0, tolerance=eps, lasso_duality_gap=oracles.lasso_duality_gap, trace=True)
                hists[lab] = hist
            else:
                oracle = orac(A, b, r)
                x_star, msg, hist = method(oracle, x_0, max_iter=10**4, tolerance=eps, trace=True)
                hists[lab] = hist

        plt.clf()
        for lab, color in zip(["sub", "prox", "bar"], ["green", "blue", "red"]):
            plt.plot(range(len(hists[lab]['duality_gap'])), np.log(hists[lab]['duality_gap']), color=color)
        plt.xlabel("Iterations")
        plt.ylabel("Log of Gap")
        plt.savefig("exp3/iter-{}-{}".format(eps_name, n))

        plt.clf()
        for lab, color in zip(["sub", "prox", "bar"], ["green", "blue", "red"]):
            plt.plot(hists[lab]['time'], np.log(hists[lab]['duality_gap']), color=color)
        plt.xlabel("Seconds")
        plt.ylabel("Log of Gap")
        plt.savefig("exp3/second-{}-{}".format(eps_name, n))
