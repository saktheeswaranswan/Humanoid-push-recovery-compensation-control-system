from logging import getLogger

import numpy as np
import scipy.stats as stats

from .controller import Controller
from ..common.utils import line_search

logger = getLogger(__name__)


class NMPC(Controller):
    def __init__(self, model):
        """ Nonlinear Model Predictive Control using pure gradient algorithm
        """
        super(NMPC, self).__init__(model)

        # model
        self.model = model

        # general parameters
        self.pred_len = 50
        self.input_size = self.model.nu
        self.dt = 0.02

        self.Q = np.eye(1)
        self.Q_f = np.eye(1)
        self.R = np.eye(1)

        # get cost func
        self.state_cost_fn = self.state_cost_fn
        self.terminal_state_cost_fn = self.terminal_state_cost_fn
        self.input_cost_fn = self.input_cost_fn

        # controller parameters
        self.threshold = 0.01
        self.max_iters = 5000
        self.learning_rate = 0.01
        self.optimizer_mode = "conjugate"

        self.INPUT_LOWER_BOUND = 0
        
        # initialize
        self.prev_sol = np.zeros((self.pred_len, self.input_size))

    def input_cost_fn(self, u):
        """ input cost functions

        Args:
            u (numpy.ndarray): input, shape(pred_len, input_size)
                or shape(pop_size, pred_len, input_size)
        Returns:
            cost (numpy.ndarray): cost of input, shape(pred_len, input_size) or
                shape(pop_size, pred_len, input_size)
        """
        return (u**2) * np.diag(self.R)

    def state_cost_fn(self, x, g_x):
        """ state cost function

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(pred_len, state_size)
                or shape(pop_size, pred_len, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, shape(pred_len, 1) or
                shape(pop_size, pred_len, 1)
        """

        return x**2 * np.diag(self.Q)

    def terminal_state_cost_fn(self, terminal_x, terminal_g_x):
        """

        Args:
            terminal_x (numpy.ndarray): terminal state,
                shape(state_size, ) or shape(pop_size, state_size)
            terminal_g_x (numpy.ndarray): terminal goal state,
                shape(state_size, ) or shape(pop_size, state_size)
        Returns:
            cost (numpy.ndarray): cost of state, shape(pred_len, ) or
                shape(pop_size, pred_len)
        """

        return terminal_x**2 * np.diag(self.Q_f)

    def gradient_cost_fn_state(self, x, g_x, terminal=False):
        """ gradient of costs with respect to the state

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(pred_len, state_size)

        Returns:
            l_x (numpy.ndarray): gradient of cost, shape(pred_len, state_size)
                or shape(1, state_size)
        """
        if not terminal:
            return 2. * x * np.diag(self.Q)

        return 2. * x.reshape([1,6]) * np.diag(self.Q_f)

    def gradient_cost_fn_input(self, x, u):
        """ gradient of costs with respect to the input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)
        Returns:
            l_u (numpy.ndarray): gradient of cost, shape(pred_len, input_size)
        """
        return 2. * u * np.diag(self.R)

    def hessian_cost_fn_state(x, g_x, terminal=False):
        """ hessian costs with respect to the state

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            g_x (numpy.ndarray): goal state, shape(pred_len, state_size)
        Returns:
            l_xx (numpy.ndarray): gradient of cost,
                shape(pred_len, state_size, state_size) or
                shape(1, state_size, state_size) or
        """
        if not terminal:
            (pred_len, state_size) = x.shape
            hessian = np.eye(state_size)*2
            hessian = np.tile(hessian, (pred_len, 1, 1))

            return hessian

        state_size = len(x)
        hessian = np.eye(state_size)*2

        return hessian[np.newaxis, :, :]

    def hessian_cost_fn_input(self, x, u):
        """ hessian costs with respect to the input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)
        Returns:
            l_uu (numpy.ndarray): gradient of cost,
                shape(pred_len, input_size, input_size)
        """
        (pred_len, _) = u.shape

        return np.tile(2.*self.R, (pred_len, 1, 1))

    def hessian_cost_fn_input_state(x, u):
        """ hessian costs with respect to the state and input

        Args:
            x (numpy.ndarray): state, shape(pred_len, state_size)
            u (numpy.ndarray): goal state, shape(pred_len, input_size)
        Returns:
            l_ux (numpy.ndarray): gradient of cost ,
                shape(pred_len, input_size, state_size)
        """
        (_, state_size) = x.shape
        (pred_len, input_size) = u.shape

        return np.zeros((pred_len, input_size, state_size))

    def gradient_hamiltonian_input(self, x, lam, u, g_x, dummy_u, raw):
        """
        Args:
            x (numpy.ndarray): shape(pred_len+1, state_size)
            lam (numpy.ndarray): shape(pred_len, state_size)
            u (numpy.ndarray): shape(pred_len, input_size)
            g_xs (numpy.ndarray): shape(pred_len, state_size)
            dummy_u (numpy.ndarray): shape(pred_len, input_size)
            raw (numpy.ndarray): shape(pred_len, input_size), Lagrangian for constraints

        Returns:
            F (numpy.ndarray), shape(pred_len, 3)
        """
        if len(x.shape) == 1:
            vanilla_F = np.zeros(1)
            extend_F = np.zeros(1)  # 1 is the same as input size
            extend_C = np.zeros(1)

            vanilla_F[0] = u[0] + lam[1] + 2. * raw[0] * u[0]
            extend_F[0] = -0.01 + 2. * raw[0] * dummy_u[0]
            extend_C[0] = u[0]**2 + dummy_u[0]**2 - \
                self.INPUT_LOWER_BOUND**2

            F = np.concatenate([vanilla_F, extend_F, extend_C])

        elif len(x.shape) == 2:
            pred_len, _ = u.shape
            vanilla_F = np.zeros((pred_len, 1))
            extend_F = np.zeros((pred_len, 1))  # 1 is the same as input size
            extend_C = np.zeros((pred_len, 1))

            for i in range(pred_len):
                vanilla_F[i, 0] = \
                    u[i, 0] + lam[i, 1] + 2. * raw[i, 0] * u[i, 0]
                extend_F[i, 0] = -0.01 + 2. * raw[i, 0] * dummy_u[i, 0]
                extend_C[i, 0] = u[i, 0]**2 + dummy_u[i, 0]**2 - \
                    self.INPUT_LOWER_BOUND**2

            F = np.concatenate([vanilla_F, extend_F, extend_C], axis=1)

        return F


    def obtain_sol(self, curr_x, g_xs):
        """ calculate the optimal inputs
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        sol = self.prev_sol.copy()
        count = 0
        # use for Conjugate method
        conjugate_d = None
        conjugate_prev_d = None
        conjugate_s = None
        conjugate_beta = None

        while True:
            # shape(pred_len+1, state_size)
            pred_xs = self.model.predict_traj(curr_x, sol)
            # shape(pred_len, state_size)
            pred_lams = self.model.predict_adjoint_traj(pred_xs, sol, g_xs)

            F_hat = self.gradient_hamiltonian_input(
                pred_xs, pred_lams, sol, g_xs)

            if np.linalg.norm(F_hat) < self.threshold:
                break

            if count > self.max_iters:
                logger.debug(" break max iteartion at F : `{}".format(
                    np.linalg.norm(F_hat)))
                break

            if self.optimizer_mode == "conjugate":
                conjugate_d = F_hat.flatten()

                if conjugate_prev_d is None:  # initial
                    conjugate_s = conjugate_d
                    conjugate_prev_d = conjugate_d
                    F_hat = conjugate_s.reshape(F_hat.shape)
                else:
                    prev_d = np.dot(conjugate_prev_d, conjugate_prev_d)
                    d = np.dot(conjugate_d, conjugate_d - conjugate_prev_d)
                    conjugate_beta = (d + 1e-6) / (prev_d + 1e-6)

                    conjugate_s = conjugate_d + conjugate_beta * conjugate_s
                    conjugate_prev_d = conjugate_d
                    F_hat = conjugate_s.reshape(F_hat.shape)

            def compute_eval_val(u):
                pred_xs = self.model.predict_traj(curr_x, u)
                state_cost = np.sum(self.state_cost_fn(
                    pred_xs[1:-1], g_xs[1:-1]))
                input_cost = np.sum(self.input_cost_fn(u))
                terminal_cost = np.sum(
                    self.terminal_state_cost_fn(pred_xs[-1], g_xs[-1]))
                return state_cost + input_cost + terminal_cost

            alpha = line_search(F_hat, sol,
                                compute_eval_val, init_alpha=self.learning_rate)

            sol -= alpha * F_hat
            count += 1

        # update us for next optimization
        self.prev_sol = np.concatenate(
            (sol[1:], np.zeros((1, self.input_size))), axis=0)

        return sol[0]
