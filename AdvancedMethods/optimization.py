import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
import time
from numpy import linalg as li


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

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
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    iters = 0
    start = time.time()

    if max_iter is not None:
        max_iter = np.alen(b)

    def pushHistory(g_norm, x):
        if trace:
            history['time'].append(time.time() - start)
            history['residual_norm'].append(g_norm)
            if np.alen(x) <= 2:
                history['x'].append(x)
        if display:
            print(x)

    b_norm = li.norm(b)

    x_k = np.copy(x_0) * 1.0
    Ax_k = matvec(x_k)
    g_k = Ax_k - b
    d_k = g_k * (-1)
    Ad_k = matvec(d_k)

    g_k_norm = li.norm(g_k)

    while g_k_norm > b_norm * tolerance:
        if max_iter is not None and iters > max_iter:
            return x_k, 'iterations_exceeded', history
        iters += 1
        pushHistory(g_k_norm, x_k)
        koef = (g_k.dot(g_k) / d_k.dot(Ad_k))
        x_k = x_k + d_k * koef
        Ax_k = Ax_k + Ad_k * koef
        g_k_prev_square = g_k.dot(g_k)
        g_k = Ax_k - b
        g_k_norm = li.norm(g_k)
        d_k = g_k*(-1) + d_k * g_k.dot(g_k) / g_k_prev_square
        Ad_k = matvec(d_k)

    pushHistory(g_k_norm, x_k)
    return x_k, 'success', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10, 
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0) * 1.0
    x_k_old = None
    df_k_old = None
    H = deque()
    l = memory_size

    start = time.time()

    def pushHistory(x_k, oracle, df_k_n):
        if trace:
            history['time'].append(time.time() - start)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(df_k_n ** (1 / 2))
            if np.alen(x_k) <= 2:
                history['x'].append(np.copy(x_k))
        if display:
            print(x_k)

    df0 = oracle.grad(x_0)
    df0_norm = df0.dot(df0)
    for k in range(max_iter):
        if x_k_old is not None:
            df_k_old = np.copy(df_k)

        df_k = oracle.grad(x_k)
        # hf_k = oracle.hess(x_k)
        df_k_norm = df_k.dot(df_k)
        pushHistory(x_k, oracle, df_k_norm)
        if df_k_norm <= df0_norm * tolerance:
            return x_k, 'success', history

        if x_k_old is not None:
            H.append((np.copy(x_k-x_k_old), np.copy(df_k-df_k_old)))
            if len(H) > l:
                H.popleft()

        d_k = df_k * (-1.0)
        mus = list()
        for s, y in reversed(H):
            mu = s.dot(d_k) / s.dot(y)
            mus.append(mu)
            d_k = d_k - y * mu
        if len(H) > 0:
            s_k, y_k = H[-1]
            d_k = d_k * s_k.dot(y_k) / y_k.dot(y_k)
        for s_y, mu in zip(H, reversed(mus)):
            s, y = s_y
            betta = y.dot(d_k) / s.dot(y)
            d_k = d_k + s * (mu - betta)

        a_k = line_search_tool.line_search(oracle, x_k, d_k)
        x_k_old = np.copy(x_k)
        x_k += d_k * a_k

    df_last = oracle.grad(x_k)
    df_last_norm = df_last.dot(df_last)
    pushHistory(x_k, oracle, df_last_norm)
    if df_last_norm <= df0_norm * tolerance:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
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
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0) * 1.0

    start = time.time()

    teta_k = lambda n_k: min(0.5, n_k**(1/4))

    def pushHistory(x_k, oracle, df_k_n):
        if trace:
            history['time'].append(time.time() - start)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(df_k_n ** (1 / 2))
            if np.alen(x_k) <= 2:
                history['x'].append(np.copy(x_k))
        if display:
            print(x_k)

    df0 = oracle.grad(x_0)
    df0_norm = df0.dot(df0)
    for k in range(max_iter):
        df_k = oracle.grad(x_k)
        df_k_norm = df_k.dot(df_k)
        pushHistory(x_k, oracle, df_k_norm)
        if df_k_norm <= df0_norm * tolerance:
            return x_k, 'success', history

        teta = teta_k(df_k_norm)
        matvec = lambda v: oracle.hess_vec(x_k, v)
        d_k, msg, hist = conjugate_gradients(matvec, df_k*(-1), df_k*(-1), tolerance=teta)
        while d_k.dot(df_k) > 0:
            teta /= 10
            d_k, msg, hist = conjugate_gradients(matvec, df_k * (-1), df_k*(-1), tolerance=teta)


        a_k = line_search_tool.line_search(oracle, x_k, d_k)
        x_k += d_k * a_k

    df_last = oracle.grad(x_k)
    df_last_norm = df_last.dot(df_last)
    pushHistory(x_k, oracle, df_last_norm)
    if df_last_norm <= df0_norm * tolerance:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history



# ------------OLD
def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
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
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start = time.time()

    def pushHistory(x_k, oracle, df_k_n):
        if trace:
            history['time'].append(time.time()-start)
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
        df_k_norm = df_k.dot(df_k)
        pushHistory(x_k, oracle, df_k_norm)
        if df_k_norm <= df0_norm * tolerance:
            return x_k, 'success', history
        d_k = df_k*(-1)
        a_k = line_search_tool.line_search(oracle, x_k, d_k)
        x_k += d_k * a_k

    df_last = oracle.grad(x_k)
    df_last_norm = df_last.dot(df_last)
    pushHistory(x_k, oracle, df_last_norm)
    if df_last_norm <= df0_norm * tolerance:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history