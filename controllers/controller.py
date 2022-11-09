import numpy as np

class Controller():
    """ Controller class
    """

    def __init__(self, config, model):
        """
        """
        self.config = config
        self.model = model

        # get cost func
        self.state_cost_fn = config.state_cost_fn
        self.terminal_state_cost_fn = config.terminal_state_cost_fn
        self.input_cost_fn = config.input_cost_fn

    def obtain_sol(self, curr_x, g_xs):
        """ calculate the optimal inputs
        Args:
            curr_x (numpy.ndarray): current state, shape(state_size, )
            g_xs (numpy.ndarrya): goal trajectory, shape(plan_len, state_size)
        Returns:
            opt_input (numpy.ndarray): optimal input, shape(input_size, )
        """
        raise NotImplementedError("Implement the algorithm to \
                                   get optimal input")

    

    def calc_cost1(pred_xs, input_sample, g_xs,
                state_cost_fn, input_cost_fn, terminal_state_cost_fn):
        """ calculate the cost 
        Args:
            pred_xs (numpy.ndarray): predicted state trajectory, 
                shape(pop_size, pred_len+1, state_size)
            input_sample (numpy.ndarray): inputs samples trajectory,
                shape(pop_size, pred_len+1, input_size)
            g_xs (numpy.ndarray): goal state trajectory,
                shape(pop_size, pred_len+1, state_size)
            state_cost_fn (function): state cost fucntion
            input_cost_fn (function): input cost fucntion
            terminal_state_cost_fn (function): terminal state cost fucntion
        Returns:
            cost (numpy.ndarray): cost of the input sample, shape(pop_size, )
        """
        # state cost
        state_cost = 0.
        if state_cost_fn is not None:
            state_pred_par_cost = state_cost_fn(
                pred_xs[:, 1:-1, :], g_xs[:, 1:-1, :])
            state_cost = np.sum(np.sum(state_pred_par_cost, axis=-1), axis=-1)

        # terminal cost
        terminal_state_cost = 0.
        if terminal_state_cost_fn is not None:
            terminal_state_par_cost = terminal_state_cost_fn(pred_xs[:, -1, :],
                                                            g_xs[:, -1, :])
            terminal_state_cost = np.sum(terminal_state_par_cost, axis=-1)

        # act cost
        act_cost = 0.
        if input_cost_fn is not None:
            act_pred_par_cost = input_cost_fn(input_sample)
            act_cost = np.sum(np.sum(act_pred_par_cost, axis=-1), axis=-1)

        return state_cost + terminal_state_cost + act_cost

    def calc_cost(self, curr_x, samples, g_xs):
        """ calculate the cost of input samples

        Args:
            curr_x (numpy.ndarray): shape(state_size),
                current robot position
            samples (numpy.ndarray): shape(pop_size, opt_dim), 
                input samples
            g_xs (numpy.ndarray): shape(pred_len, state_size),
                goal states
        Returns:
            costs (numpy.ndarray): shape(pop_size, )
        """
        # get size
        pop_size = samples.shape[0]
        g_xs = np.tile(g_xs, (pop_size, 1, 1))

        # calc cost, pred_xs.shape = (pop_size, pred_len+1, state_size)
        pred_xs = self.model.predict_traj(curr_x, samples)

        # get particle cost
        costs = self.calc_cost1(pred_xs, samples, g_xs,
                          self.state_cost_fn, self.input_cost_fn,
                          self.terminal_state_cost_fn)

        return costs
