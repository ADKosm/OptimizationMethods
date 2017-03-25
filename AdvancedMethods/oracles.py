import numpy as np
import scipy
import scipy.sparse as sp
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return np.array(self.hess(x)).dot(v)

    def minimize_directional(self, x, d):
        raise NotImplementedError('Hessian oracle is not implemented.')


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def minimize_directional(self, x, d):
        """
        Minimizes the function with respect to a specific direction:
            Finds alpha = argmin f(x + alpha d)
        """
        return (self.b.dot(d) - np.dot(self.A.dot(x), d)) / np.dot(self.A.dot(d), d)


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATy : function of y
            Computes matrix-vector product A^Ty, where y is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.diag_b = sp.diags(b, 0)
        self.matmat_diagbA = lambda x: self.diag_b.dot(x)
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        c = np.mean(np.vectorize(lambda x: np.logaddexp(0, x))(self.matmat_diagbA(self.matvec_Ax(x)) * (-1))) + np.dot(x, x) * self.regcoef / 2
        return c

    def grad(self, x):
        m = np.alen(self.b)
        x = np.array(x)
        c = self.matvec_ATx(
            self.matmat_diagbA(
                np.vectorize(scipy.special.expit)( self.matmat_diagbA(
                        self.matvec_Ax(x)
                    ) * (-1)
                )
            )
        )*(-1/m) + x * self.regcoef
        # -1/m * AT*Diag(b)*sigma(-Diag(b)*A*x) + lambda*x
        return c

    def hess(self, x):
        m = np.alen(self.b)
        n = np.alen(x)
        c = self.matmat_ATsA(
            np.vectorize(lambda x: scipy.special.expit(x)*(1-scipy.special.expit(x)))(
                self.matmat_diagbA(
                    self.matvec_Ax(x)
                )*(-1)
            )
        )*(1/m) + np.eye(n) * self.regcoef
        return c

    def hess_vec(self, x, v):
        m = np.alen(self.b)
        n = np.alen(x)
        s = sp.diags(np.vectorize(lambda x: scipy.special.expit(x)*(1-scipy.special.expit(x)))(
                self.matmat_diagbA(
                    self.matvec_Ax(x)
                )*(-1)
            ), 0) * (1/m)
        c = self.matvec_ATx(s.dot(self.matvec_Ax(v))) + v * self.regcoef
        return c


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.memoized = {
            'a' : 0,
            'x' : None,
            'd' : None,
            'Ax': None,
            'Ad': None
        }
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def o(self, entity, rvalue):
        if np.allclose(np.zeros(rvalue.shape[0]), rvalue):
            return np.zeros(self.b.shape[0])
        if entity == 'Ad':
            if self.memoized['d'] is None:
                self.memoized['d'] = np.copy(rvalue)
                self.memoized['Ad'] = self.matvec_Ax(rvalue)
                return self.memoized['Ad']
            if np.allclose(self.memoized['d'], rvalue):
                return self.memoized['Ad']
            else:
                self.memoized['d'] = np.copy(rvalue)
                self.memoized['Ad'] = self.matvec_Ax(rvalue)
                return self.memoized['Ad']
        elif entity == 'Ax':
            if self.memoized['x'] is None:
                self.memoized['x'] = np.copy(rvalue)
                self.memoized['Ax'] = self.matvec_Ax(rvalue)
                return self.memoized['Ax']
            if np.allclose(self.memoized['x'], rvalue):
                return self.memoized['Ax']
            else:
                if np.allclose(self.memoized['x'] + self.memoized['a']*self.memoized['d'], rvalue):
                    self.memoized['x'] = np.copy(rvalue)
                    self.memoized['Ax'] = self.memoized['Ax'] + self.memoized['a']*self.memoized['Ad']
                    return self.memoized['Ax']
                else:
                    self.memoized['x'] = np.copy(rvalue)
                    self.memoized['Ax'] = self.matvec_Ax(rvalue)
                    return self.memoized['Ax']

    def dpsi(self, x, d, alpha):
        arg = self.o('Ax', x) + alpha * self.o('Ad', d)
        m = np.alen(self.b)
        x = np.array(x)
        c = self.matmat_diagbA(
            np.vectorize(scipy.special.expit)(self.matmat_diagbA(
                arg
            ) * (-1)
                                              )
        ) * (-1 / m)
        self.memoized['a'] = alpha
        return c

    def func(self, x):
        return self.func_directional(x, np.zeros(x.shape[0]), 0)

    def grad(self, x):
        return self.matvec_ATx(self.dpsi(x, np.zeros(x.shape[0]), 0)) + x * self.regcoef

    def hess(self, x):
        m = np.alen(self.b)
        n = np.alen(x)
        c = self.matmat_ATsA(
            np.vectorize(lambda x: scipy.special.expit(x) * (1 - scipy.special.expit(x)))(
                self.matmat_diagbA(
                    self.o('Ax', x)
                ) * (-1)
            )
        ) * (1 / m) + np.eye(n) * self.regcoef
        return c

    def hess_vec(self, x, v):
        m = np.alen(self.b)
        s = sp.diags(np.vectorize(lambda x: scipy.special.expit(x)*(1-scipy.special.expit(x)))(
                self.matmat_diagbA(
                    self.o('Ax', x)
                )*(-1)
            ), 0) * (1/m)
        c = self.matvec_ATx(s.dot(self.matvec_Ax(v))) + v * self.regcoef
        return c

    def func_directional(self, x, d, alpha):
        arg = self.o('Ax', x) + alpha * self.o('Ad', d)
        c = np.mean(np.vectorize(lambda x: np.logaddexp(0, x))(self.matmat_diagbA(arg) * (-1))) + np.dot(x + d * alpha, x + d * alpha) * self.regcoef / 2
        self.memoized['a'] = alpha
        return np.squeeze(c)

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.dpsi(x, d, alpha).dot(self.o('Ad', d)) + (x + d * alpha).dot(d) * self.regcoef)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    def matmat_ATsA(s):
        return A.T.dot(sp.diags(s, 0).dot(A))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """
    # TODO: Implement numerical estimation of the Hessian times vector
    n = np.alen(x)
    E = np.eye(n)
    hess = np.array([(func(x + v * eps + e_j * eps) - func(x + v * eps) - func(x + e_j * eps) + func(x)) / eps ** 2
                     for e_j in E])
    return hess