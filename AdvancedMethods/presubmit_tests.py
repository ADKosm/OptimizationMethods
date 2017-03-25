import sys
from io import StringIO
import unittest
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_array_almost_equal

from oracles import QuadraticOracle, create_log_reg_oracle, hess_vec_finite_diff
from optimization import conjugate_gradients, lbfgs, hessian_free_newton
from utils import LineSearchTool

# Check if it's Python 3
if not sys.version_info > (3, 0):
    print('You should use only Python 3!')
    sys.exit()

test_bonus = 'bonus' in sys.argv

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


class TestCG(unittest.TestCase):
    # Define a simple linear system with A = A' > 0
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    x0 = np.array([0, 0])
    matvec = (lambda self, x: self.A.dot(x))

    def check_equal_histories(self, history1, history2, atol=1e-3):
        if history1 is None or history2 is None:
            self.assertEqual(history1, history2)
            return
        self.assertTrue('residual_norm' in history1 and 'residual_norm' in history2)
        self.assertTrue(np.allclose(history1['residual_norm'], history2['residual_norm'], atol=atol))
        self.assertTrue('time' in history1 and 'time' in history2)
        self.assertTrue(len(history1['time']), len(history2['time']))
        self.assertTrue('x' in history1, 'x' in history2)
        if 'x' in history1:
            self.assertTrue(np.allclose(history1['x'], history2['x'], atol=atol))

    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            x_sol, message, _ = conjugate_gradients(self.matvec, self.b, self.x0)

        assert_equal(message, 'success')
        g_k_norm = norm(self.A.dot(x_sol) - self.b, 2)
        self.assertLessEqual(g_k_norm, 1e-4 * norm(self.b))
        self.assertEqual(len(output), 0, 'You should not print anything by default.')

    def test_tol(self):
        """Check if argument `tol` is supported."""
        conjugate_gradients(self.matvec, self.b, self.x0, tolerance=1e-6)

    def test_max_iter(self):
        """Check argument `max_iter` is supported and can be set to None."""
        conjugate_gradients(self.matvec, self.b, self.x0, max_iter=None)

    def test_disp(self):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            conjugate_gradients(self.matvec, self.b, self.x0, display=True)

        self.assertLess(0, len(output), 'You should print the progress when `display` is True.')

    def test_trace(self):
        """Check if the history is returned correctly when `trace` is True."""
        x_sol, message, hist = conjugate_gradients(self.matvec, self.b, self.x0, trace=True)

        self.assertTrue(isinstance(hist['residual_norm'], list) or isinstance(hist['residual_norm'], np.ndarray))
        self.assertEqual(len(hist['residual_norm']), len(hist['time']))
        self.assertEqual(len(hist['residual_norm']), len(hist['x']))

        x_sol, message, hist = conjugate_gradients(self.matvec, self.b, self.x0, trace=False)
        self.assertTrue(hist is None)


    def test_histories(self):
        x_sol, message, history = conjugate_gradients(self.matvec, self.b, self.x0, trace=True)
        res_norm = [6.0827625302982193,
                    0.49995308468204547,
                    0.0]
        time_steps = [0.0] * 3  # Dummy values
        x_steps = [np.array([0, 0]),
                   np.array([ 0.50684932,  3.04109589]),
                   np.array([ 1.,  3.])
                   ]
        true_history = dict(residual_norm=res_norm, time=time_steps, x=x_steps)
        self.check_equal_histories(history, true_history)


class TestLBFGS(unittest.TestCase):
    # Define a simple quadratic function for testing
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    oracle = QuadraticOracle(A, b)

    f_star = -9.5
    x0 = np.array([0, 0])
    # For this func |nabla f(x)| < tol ensures |f(x) - f(x^*)| < tol^2

    def check_equal_histories(self, history1, history2, atol=1e-3):
        if history1 is None or history2 is None:
            self.assertEqual(history1, history2)
            return
        self.assertTrue('func' in history1 and 'func' in history2)
        self.assertTrue(np.allclose(history1['func'], history2['func'], atol=atol))
        self.assertTrue('grad_norm' in history1 and 'grad_norm' in history2)
        self.assertTrue(np.allclose(history1['grad_norm'], history2['grad_norm'], atol=atol))
        self.assertTrue('time' in history1 and 'time' in history2)
        self.assertTrue(len(history1['time']), len(history2['time']))
        self.assertTrue('x' in history1, 'x' in history2)
        if 'x' in history1:
            self.assertTrue(np.allclose(history1['x'], history2['x'], atol=atol))

    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            x_min, message, _ = lbfgs(self.oracle, self.x0)

        assert_equal(message, 'success')
        self.assertEqual(len(output), 0, 'You should not print anything by default.')

    def test_tol(self):
        """Check if argument `tol` is supported."""
        lbfgs(self.oracle, self.x0, tolerance=1e-6)

    def test_max_iter(self):
        """Check if argument `max_iter` is supported."""
        lbfgs(self.oracle, self.x0, max_iter=15)

    def test_memory_size(self):
        """Check if argument `memory_size` is supported."""
        lbfgs(self.oracle, self.x0, memory_size=1)

    def test_line_search_options(self):
        """Check if argument `line_search_options` is supported."""
        lbfgs(self.oracle, self.x0, line_search_options={'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9})

    def test_disp(self):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            lbfgs(self.oracle, self.x0, display=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `display` is True.')

    def test_trace(self):
        """Check if the history is returned correctly when `trace` is True."""
        x_min, message, hist = lbfgs(self.oracle, self.x0, trace=True)

        self.assertEqual(len(hist['grad_norm']), len(hist['func']))
        self.assertEqual(len(hist['time']), len(hist['func']))
        self.assertEqual(len(hist['x']), len(hist['func']))

        x_min, message, hist = lbfgs(self.oracle, self.x0, trace=False)
        self.assertTrue(hist is None)


    def test_quality(self):
        x_min, message, _ = lbfgs(self.oracle, self.x0, tolerance=1e-10)
        f_min = self.oracle.func(x_min)

        g_k_norm = norm(self.A.dot(x_min) - self.b, 2)
        self.assertLessEqual(abs(g_k_norm), 1e-3)
        self.assertLessEqual(abs(f_min - self.f_star), 1e-3)

    def test_histories(self):
        x0 = -np.array([1.3, 2.7])
        x_min, message, history = lbfgs(self.oracle, x0, trace=True, memory_size=10, line_search_options={'method': 'Constant', 'c': 1.0},
                                        tolerance=1e-6)
        func_steps = [25.635000000000005,
                      22.99,
                      -9.3476294733722725,
                      -9.4641732176886055,
                      -9.5]
        grad_norm_steps = [11.629703349613008,
                           11.4,
                           0.55751193505619512,
                           0.26830541958992876,
                           0.0]
        time_steps = [0.0] * 5  # Dummy values
        x_steps = [np.array([-1.3, -2.7]),
                   np.array([1.0,  8.7]),
                   np.array([0.45349973,  3.05512941]),
                   np.array([0.73294321,  3.01292737]),
                   np.array([0.99999642,  2.99998814])]
        true_history = dict(grad_norm=grad_norm_steps, time=time_steps, x=x_steps, func=func_steps)
        self.check_equal_histories(history, true_history)
        self.assertEqual(message, "success")
        assert_array_almost_equal(x_min, np.array([1.0, 3.0]), decimal=3)


class TestHFN(unittest.TestCase):
    # Define a simple quadratic function for testing
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    f_star = -9.5
    x0 = np.array([0, 0])
    # no need for `extra` for this simple function
    oracle = QuadraticOracle(A, b)
    # For this func |nabla f(x)| < tol ensures |f(x) - f(x^*)| < tol^2

    def check_equal_histories(self, history1, history2, atol=1e-3):
        if history1 is None or history2 is None:
            self.assertEqual(history1, history2)
            return
        self.assertTrue('func' in history1 and 'func' in history2)
        self.assertTrue(np.allclose(history1['func'], history2['func'], atol=atol))
        self.assertTrue('grad_norm' in history1 and 'grad_norm' in history2)
        self.assertTrue(np.allclose(history1['grad_norm'], history2['grad_norm'], atol=atol))
        self.assertTrue('time' in history1 and 'time' in history2)
        self.assertTrue(len(history1['time']), len(history2['time']))
        self.assertTrue('x' in history1, 'x' in history2)
        if 'x' in history1:
            self.assertTrue(np.allclose(history1['x'], history2['x'], atol=atol))

    def test_default(self):
        """Check if everything works correctly with default parameters."""
        with Capturing() as output:
            x_min, message, _ = hessian_free_newton(self.oracle, self.x0)

        assert_equal(message, 'success')
        self.assertTrue(len(output) == 0, 'You should not print anything by default.')

    def test_tol(self):
        """Check if argument `tol` is supported."""
        hessian_free_newton(self.oracle, self.x0, tolerance=1e-6)

    def test_max_iter(self):
        """Check if argument `max_iter` is supported."""
        hessian_free_newton(self.oracle, self.x0, max_iter=15)

    def test_line_search_options(self):
        """Check if argument `line_search_options` is supported."""
        hessian_free_newton(self.oracle, self.x0, line_search_options={'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9})

    def test_disp(self):
        """Check if something is printed when `disp` is True."""
        with Capturing() as output:
            hessian_free_newton(self.oracle, self.x0, display=True)

        self.assertTrue(len(output) > 0, 'You should print the progress when `display` is True.')

    def test_trace(self):
        """Check if the history is returned correctly when `trace` is True."""
        x_min, message, hist = hessian_free_newton(self.oracle, self.x0, trace=True)

        self.assertEqual(len(hist['grad_norm']), len(hist['func']))
        self.assertEqual(len(hist['time']), len(hist['func']))
        self.assertEqual(len(hist['x']), len(hist['func']))

        x_min, message, hist = hessian_free_newton(self.oracle, self.x0, trace=False)
        self.assertTrue(hist is None)

    def test_quality(self):
        x_min, message, _ = hessian_free_newton(self.oracle, self.x0, tolerance=1e-10)
        f_min = self.oracle.func(x_min)

        g_k_norm = norm(self.A.dot(x_min) - self.b, 2)
        self.assertLessEqual(abs(g_k_norm), 1e-3)
        self.assertLessEqual(abs(f_min - self.f_star), 1e-3)

    def test_histories(self):
        x0 = -np.array([1.3, 2.7])
        x_min, message, history = hessian_free_newton(self.oracle, x0, trace=True, line_search_options={'method': 'Constant', 'c': 1.0},
                                                      tolerance=1e-6)

        func_steps = [25.635, -9.5]
        grad_norm_steps = [11.629703349613008, 0.0]
        time_steps = [0.0] * 2  # Dummy values
        x_steps = [np.array([-1.3, -2.7]),
                   np.array([1., 3.])]
        true_history = dict(grad_norm=grad_norm_steps, time=time_steps, x=x_steps, func=func_steps)
        self.check_equal_histories(history, true_history)
        self.assertEqual(message, "success")
        assert_array_almost_equal(x_min, np.array([1.0, 3.0]), decimal=3)


class TestHessVec(unittest.TestCase):
    # Define a simple quadratic function for testing
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    x0 = np.array([0, 0])
    # no need for `extra` for this simple function
    oracle = QuadraticOracle(A, b)

    def test_hess_vec(self):
        # f(x, y) = x^3 + y^2
        func = lambda x: x[0] ** 3 + x[1] ** 2
        x = np.array([2.0, 3.0])
        v = np.array([1.0, 0.1])
        hess_vec_test = hess_vec_finite_diff(func, x, v, eps=1e-5)
        hess_vec_real = np.array([12, 0.2])
        assert_array_almost_equal(hess_vec_real, hess_vec_test, decimal=3)

        v = np.array([1.0, -0.1])
        hess_vec_test = hess_vec_finite_diff(func, x, v, eps=1e-5)
        hess_vec_real = np.array([12, -0.2])
        assert_array_almost_equal(hess_vec_real, hess_vec_test, decimal=3)


@unittest.skipUnless(test_bonus, 'Skipping bonus tests...')
class TestBonus(unittest.TestCase):
    # Define a simple quadratic function for testing
    A = np.array([[1, 0], [0, 2]])
    b = np.array([1, 6])
    x0 = np.array([0, 0])
    # no need for `extra` for this simple function
    oracle = QuadraticOracle(A, b)

    def check_equal_histories(self, history1, history2, atol=1e-3):
        if history1 is None or history2 is None:
            self.assertEqual(history1, history2)
            return
        self.assertTrue('func' in history1 and 'func' in history2)
        self.assertTrue(np.allclose(history1['func'], history2['func'], atol=atol))
        self.assertTrue('grad_norm' in history1 and 'grad_norm' in history2)
        self.assertTrue(np.allclose(history1['grad_norm'], history2['grad_norm'], atol=atol))
        self.assertTrue('time' in history1 and 'time' in history2)
        self.assertTrue(len(history1['time']), len(history2['time']))
        self.assertTrue('x' in history1, 'x' in history2)
        if 'x' in history1:
            self.assertTrue(np.allclose(history1['x'], history2['x'], atol=atol))

    def test_line_search(self):
        ls_tool = LineSearchTool(method='Best')
        x_k = np.array([2.0, 2.0])
        d_k = np.array([-1.0, 1.0])
        alpha_test = ls_tool.line_search(self.oracle, x_k, d_k)
        alpha_real = 1.0
        self.assertAlmostEqual(alpha_real, alpha_test)

        x_k = np.array([2.0, 2.0])
        d_k = np.array([-1.0, 0.0])
        alpha_test = ls_tool.line_search(self.oracle, x_k, d_k)
        alpha_real = 1.0
        self.assertAlmostEqual(alpha_real, alpha_test)

        x_k = np.array([10.0, 10.0])
        d_k = np.array([-1.0, -1.0])
        alpha_test = ls_tool.line_search(self.oracle, x_k, d_k)
        alpha_real = 7.666666666666667
        self.assertAlmostEqual(alpha_real, alpha_test)

    def test_lbfgs_history(self):
        x0 = -np.array([1.3, 2.7])
        x_min, message, history = lbfgs(self.oracle, x0, trace=True, memory_size=10, line_search_options={'method': 'Best'}, tolerance=1e-6)
        func_steps = [25.635000000000005,
                      -8.8519395950378961,
                      -9.5]
        grad_norm_steps = [11.629703349613008,
                           1.1497712070693149,
                           0]
        time_steps = [0.0] * 3  # Dummy values
        x_steps = [np.array([-1.3, -2.7]),
                   np.array([-0.12706157,  3.11369481]),
                   np.array([1.,  3.])]
        true_history = dict(grad_norm=grad_norm_steps, time=time_steps, x=x_steps, func=func_steps)
        self.check_equal_histories(history, true_history)
        self.assertEqual(message, "success")
        assert_array_almost_equal(x_min, np.array([1.0, 3.0]), decimal=3)

    def test_hfn_history(self):
        x0 = -np.array([1.3, 2.7])
        x_min, message, history = hessian_free_newton(self.oracle, x0, trace=True, line_search_options={'method': 'Best'}, tolerance=1e-6)

        func_steps = [25.635, -9.5]
        grad_norm_steps = [11.629703349613008, 0.0]
        time_steps = [0.0] * 2  # Dummy values
        x_steps = [np.array([-1.3, -2.7]),
                   np.array([1., 3.])]
        true_history = dict(grad_norm=grad_norm_steps, time=time_steps, x=x_steps, func=func_steps)
        self.check_equal_histories(history, true_history)
        self.assertEqual(message, "success")
        assert_array_almost_equal(x_min, np.array([1.0, 3.0]), decimal=3)


if __name__ == '__main__':
    argv = sys.argv
    if 'bonus' in argv:
        argv.remove('bonus')
    unittest.main(argv=argv)