from collections import defaultdict
import numpy as np
from numpy.linalg import norm, solve
import scipy.linalg as sli
import scipy.optimize.linesearch as ls
import time
import datetime


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """

        phi = lambda a: oracle.func_directional(x_k, d_k, a)
        dphi = lambda a: oracle.grad_directional(x_k, d_k, a)
        if self._method != 'Constant':
            alpha = self.alpha_0 if previous_alpha is None else previous_alpha

        def backtracking(a_0):
            phi_0 = phi(0)
            dphi_0 = dphi(0)
            while phi(a_0) > phi_0 + self.c1 * a_0 * dphi_0:
                a_0 /= 2
            return a_0

        sp = lambda x: np.array_split(x, 2)  # x -> x[:half], x[half:]

        y_up, y_down = sp(x_k)
        d_up, d_down = sp(d_k)

        try: alpha_max_1 = np.min( ((y_up - y_down) / (d_down - d_up))[d_down - d_up < 0] )
        except: alpha_max_1 = self.alpha_0

        try: alpha_max_2 = np.min( (-(y_up + y_down) / (d_up + d_down))[d_down + d_up < 0] )
        except: alpha_max_2 = self.alpha_0

        alpha = min(alpha, alpha_max_1 * 0.99, alpha_max_2 * 0.99)

        if self._method == 'Armijo':
            return backtracking(alpha)
        elif self._method == 'Wolfe':
            a_wolf, b, bb, bbb = ls.scalar_search_wolfe2(phi, derphi=dphi, c1=self.c1, c2=self.c2)
            if a_wolf is None:
                return backtracking(alpha)
            else:
                return a_wolf
        elif self._method == 'Constant':
            return self.c
        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix)
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start = time.time()

    def pushHistory(x_k, oracle, df_k_n):
        if trace:
            history['time'].append(time.time() - start)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(df_k_n ** (1/2))
            if np.alen(x_k) <= 2:
                history['x'].append(np.copy(x_k))
        if display:
            print(x_k)

    df0 = oracle.grad(x_0)
    df0_norm = df0.dot(df0)
    for k in range(max_iter):
        df_k = oracle.grad(x_k)
        hf_k = oracle.hess(x_k)
        df_k_norm = df_k.dot(df_k)
        pushHistory(x_k, oracle, df_k_norm)
        if df_k_norm <= df0_norm * tolerance:
            return x_k, 'success', history
        try:
            d_k = sli.cho_solve(sli.cho_factor(hf_k), df_k*(-1))
        except sli.LinAlgError as e:
            return x_k, 'newton_direction_error', history
        a_k = line_search_tool.line_search(oracle, x_k, d_k)
        x_k += d_k * a_k

    df_last = oracle.grad(x_k)
    df_last_norm = df_last.dot(df_last)
    pushHistory(x_k, oracle, df_last_norm)
    if df_last_norm <= df0_norm * tolerance:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5, 
                         tolerance_inner=1e-8, max_iter=100, 
                         max_iter_inner=20, t_0=1, gamma=10, 
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is decreased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(ATAx_b, Ax_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary 
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    sp = lambda x: np.array_split(x, 2) # x -> x[:half], x[half:]
    me = lambda x, u: np.concatenate((x, u))
    #         | x_1 |
    #         | ... |
    # x, u -> | x_k |
    #         | u_1 |
    #         | ... |
    #         | u_k |
    matme = lambda A, B, C, D: np.vstack((np.hstack((A, B)), np.hstack((C, D))))
    # A, B, C, D ->  | A B |
    #                | C D |
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    class LassoSmoothOracle(object):
        def __init__(self, _sp, _me, _matme, _matvex_Ax, _matvec_ATx, _t, _b, _regcoef):
            self.sp = _sp
            self.me = _me
            self.matme = _matme
            self.matvec_Ax = _matvex_Ax
            self.matvec_ATx = _matvec_ATx
            self.t = _t
            self.b = _b
            self.regcoef = _regcoef

        def func(self, xx):
            x, u = self.sp(xx)
            Ax_b = self.matvec_Ax(x) - self.b
            return (Ax_b.dot(Ax_b) * 0.5 + np.sum(u) * self.regcoef) * self.t - np.sum(np.log(u-x)) - np.sum(np.log(u+x))

        def grad(self, xx):
            x, u = self.sp(xx)
            rev_op = np.vectorize(lambda s: 1/s)
            u_plus_x = rev_op(u+x)
            u_minus_x = rev_op(u-x)
            dphi_x = self.matvec_ATx(self.matvec_Ax(x)-self.b) * self.t + u_minus_x - u_plus_x
            dphi_u = np.ones(u.shape) * self.regcoef * self.t - u_minus_x - u_plus_x
            return self.me(dphi_x, dphi_u)

        def hess(self, xx):
            x, u = self.sp(xx)
            rev2_op = np.vectorize(lambda s: -1/s**2)
            u_minus_x = rev2_op(u-x)
            u_plus_x = rev2_op(u+x)
            H_xx = self.matvec_ATx(A)*self.t - np.diag(u_minus_x + u_plus_x)
            H_xu = np.diag(u_minus_x - u_plus_x)
            H_uu = np.diag(u_plus_x + u_minus_x) * (-1)
            return self.matme(H_xx, H_xu, H_xu, H_uu)

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

        def duality_gap(self, xx):
            x, u = self.sp(xx)
            Ax_b = self.matvec_Ax(x) - self.b
            ATAx_b = self.matvec_ATx(Ax_b)
            b = self.b
            regcoef = self.regcoef
            return lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)

    history = defaultdict(list) if trace else None
    x_k = np.copy(me(x_0, u_0)) * 1.0
    t_k = np.copy(t_0)

    start = time.time()

    def pushHistory(x_k, f_k, gap):
        if trace:
            history['time'].append(time.time() - start)
            history['func'].append(f_k)
            history['duality_gap'].append(gap)
            if np.alen(x_k) <= 2:
                history['x'].append(np.copy(x_k))
        if display:
            print(x_k)

    for k in range(max_iter):
        if k > max_iter:
            return tuple(sp(x_k)), 'iterations_exceeded', history
        oracle = LassoSmoothOracle(sp, me, matme, matvec_Ax, matvec_ATx, t_k, b, reg_coef)
        f_k = oracle.func(x_k)
        dual_gap = oracle.duality_gap(x_k)
        pushHistory(x_k, f_k, dual_gap)
        if dual_gap < tolerance:
            return tuple(sp(x_k)), 'success', history

        x_k, msg, hist = newton(oracle, x_k, tolerance=tolerance_inner, max_iter=max_iter_inner,
                                line_search_options={'method': "Armijo", 'c': c1})

        t_k *= gamma

    dual_gap = oracle.duality_gap(x_k)
    pushHistory(x_k, f_k, dual_gap)
    if dual_gap < tolerance:
        return tuple(sp(x_k)), 'success', history
    else:
        return tuple(sp(x_k)), 'iterations_exceeded', history

    # TODO: implement.


def subgradient_method(oracle, x_0, tolerance=1e-2, max_iter=1000, alpha_0=1,
                       display=False, trace=False):
    """
    Subgradient descent method for nonsmooth convex optimization.

    Parameters
    ----------
    oracle : BaseNonsmoothConvexOracle-descendant object
        Oracle with .func() and .subgrad() methods implemented for computing
        function value and its one (arbitrary) subgradient respectively.
        If avaliable, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    alpha_0 : float
        Initial value for the sequence of step-sizes.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0) * 1.0
    f_k = oracle.func(x_k)
    minF_x = np.copy(f_k)
    minX_k = np.copy(x_k)
    alpha_k = lambda k: alpha_0 / (k + 1) ** (0.5)
    start = time.time()

    def pushHistory(x_k, f_k, gap):
        if trace:
            history['time'].append(time.time() - start)
            history['func'].append(f_k)
            history['duality_gap'].append(gap)
            if np.alen(x_k) <= 2:
                history['x'].append(np.copy(x_k))
        if display:
            print(x_k)

    for k in range(max_iter):
        if k > max_iter:
            return x_k, 'iterations_exceeded', history
        sub_df_k = oracle.subgrad(x_k)
        sub_df_k = sub_df_k / norm(sub_df_k)
        dual_gap = oracle.duality_gap(x_k)
        pushHistory(x_k, f_k, dual_gap)
        if dual_gap < tolerance:
            return minX_k, 'success', history
        x_k -= sub_df_k * alpha_k(k)
        f_k = oracle.func(x_k)
        if f_k < minF_x:
            minF_x = np.copy(f_k)
            minX_k = np.copy(x_k)

    dual_gap = oracle.duality_gap(x_k)
    pushHistory(x_k, f_k, dual_gap)
    if dual_gap < tolerance:
        return minX_k, 'success', history
    else:
        return minX_k, 'iterations_exceeded', history


def proximal_gradient_descent(oracle, x_0, L_0=1, tolerance=1e-5,
                              max_iter=1000, trace=False, display=False):
    """
    Proximal gradient descent for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented 
        for computing function value, its gradient and proximal mapping 
        respectively.
        If avaliable, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_counter = 0
    x_k = np.copy(x_0) * 1.0
    L_k = L_0

    f_k = None
    df_k = None

    m_l = lambda y, x: f_k + df_k.dot(y-x) + L_k / 2 * (y-x).dot(y-x) + oracle._h.func(y)

    start = time.time()

    def pushHistory(x_k, oracle, gap):
        if trace:
            history['time'].append(time.time() - start)
            history['func'].append(oracle.func(x_k))
            history['duality_gap'].append(gap)
            history['line_counter'].append(np.copy(line_counter))
            if np.alen(x_k) <= 2:
                history['x'].append(np.copy(x_k))
        if display:
            print(x_k)

    for k in range(max_iter):
        if k > max_iter:
            return x_k, 'iterations_exceeded', history

        f_k = oracle._f.func(x_k)
        df_k = oracle.grad(x_k)

        dual_gap = oracle.duality_gap(x_k)
        pushHistory(x_k, oracle, dual_gap)
        if dual_gap < tolerance:
            return x_k, 'success', history

        while True:
            line_counter += 1
            y = oracle.prox(x_k - 1/L_k * df_k, 1/L_k)
            if oracle.func(y) <= m_l(y, x_k):
                break
            L_k *= 2
        x_k = np.copy(y)
        L_k = max(L_0, L_k / 2)

    dual_gap = oracle.duality_gap(x_k)
    pushHistory(x_k, oracle, dual_gap)
    if dual_gap < tolerance:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history

