# File containing all the functions for running experiments with the passive or active learner

# Add paths
import os
import numpy as np
import scipy.linalg
import scipy.stats
import matplotlib.pyplot as plt
import functools

import pickle

# Hamiltonian Learning
from ..quantum_system_oracles import process_data, simulate_nature
from ..quantum_system_models import quantum_device_models
from ..learners import design_experiment

# estimators
from ..estimators import estimation_procedure, initial_estimators, mle_estimators


# Helpful Global parameters/constants
def kron(a, b):
    return np.matrix(scipy.linalg.kron(a, b))


si = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])


class Learning_Experiment_Runner(object):
    """
    Setting up a template and/or default for learning experiments that we will be running
    """
    def __init__(self, env_cr, query_space, est_param_info, active_learner,
                 FLAG_query_constraints=False, query_constraints_info=None,
                 FLAG_adaptive_query_space=False, adaptive_query_space_info=None,
                 estimation_strategy=None,
                 N_0=972, N_batch=486,
                 FLAG_debug_AL=False):

        # Parameters relevant to environment
        # Should create an environment (simulated) using the true value of J
        # to pass to sample_action_space for querying and creating the dataset
        J_truth = np.copy(env_cr.J)
        self.env_cr = env_cr
        self.J_truth = J_truth

        # Normalization factors for error computation
        self.xi_J_error = 1e6 * np.ones(len(J_truth))

        # Extract parameters relevant to the query space
        self.init_query_space = query_space
        self.moset = query_space['moset']
        self.prepset = query_space['prepset']
        self.time_stamps = query_space['time_stamps'] # This will be changed over the course of the simulation

        # Parameters relevant to parameterization and estimation
        self.freq_convert = est_param_info['freq_convert']
        self.xi_t = est_param_info['xi_t']
        self.xi_J = est_param_info['xi_J']

        self.time_stamps_nd = self.time_stamps / self.xi_t

        if est_param_info['init_J'] is not None:
            self.FLAG_initial_estimate = self.est_param_info['FLAG_initial_estimate']
            self.init_J_num = self.est_param_info['init_J']
        else:
            if FLAG_debug_AL:
                self.init_J_num = J_truth
                self.FLAG_initial_estimate = False
            else:
                self.init_J_num = None
                self.FLAG_initial_estimate = True

        self.FLAG_debug_AL = FLAG_debug_AL

        # Parameters relevant for sampling queries
        self.N_0 = N_0
        self.N_batch = N_batch

        # Estimation strategy
        if estimation_strategy is None:
            self.estimation_strategy = {'baseline': False,
                                        'quick_MLE': True,
                                        'FLAG_initial_estimate': False,
                                        'FLAG_FFT_high_resolution': False,
                                        'FLAG_MLE_param': False,
                                        'FLAG_MLE_J': False}
        else:
            self.estimation_strategy = estimation_strategy

        # Parameters relevant to query optimization or query constraints in AL procedure
        self.FLAG_query_constraints = FLAG_query_constraints
        if FLAG_query_constraints:
            # TODO: Make sure the query constraints being fed in match the oracle's query constraints
            # N_shots can't be np.inf or np.nan as the array is later changed to an array of integers.
            # Making it a large value for now!
            default_query_constraints_info = {'query_constraints_ref': {'N_shots': 1e8},
                                              'query_optimization_type': 'batch',
                                              'max_iter': 40}

            if query_constraints_info is None:
                self.query_constraints_ref = default_query_constraints_info['query_constraints_ref']
                self.query_optimization_type = default_query_constraints_info['query_optimization_type']
                self.max_iter = default_query_constraints_info['max_iter']
            else:
                if 'query_constraints_ref' in query_constraints_info.keys():
                    self.query_constraints_ref = query_constraints_info['query_constraints_ref']
                else:
                    self.query_constraints_ref = default_query_constraints_info['query_constraints_ref']

                if 'query_optimization_type' in query_constraints_info.keys():
                    self.query_optimization_type = query_constraints_info['query_optimization_type']
                else:
                    self.query_optimization_type = default_query_constraints_info['query_optimization_type']

                if 'max_iter' in query_constraints_info.keys():
                    self.max_iter = query_constraints_info['max_iter']
                else:
                    self.max_iter = default_query_constraints_info['max_iter']

            if self.query_constraints_ref is not None:
                if 'N_shots' in self.query_constraints_ref.keys():
                    self._max_shots_ActionSpace = self.query_constraints_ref['N_shots']
            else:
                self._max_shots_ActionSpace = 1e8
        else:
            # Number of shots in the ActionSpace for each query if no query constraints present
            self._max_shots_ActionSpace = 1e8

        # Parameters relevant to Adaptive Query Space of AL
        self.FLAG_adaptive_query_space = FLAG_adaptive_query_space
        if FLAG_adaptive_query_space:
            default_adaptive_query_space_info = {'growth_time_stamps': None,
                                                 'max_iter_growth': 10,
                                                 'growth_factor': 2}

            if adaptive_query_space_info is None:
                self.adaptive_qs_info = default_adaptive_query_space_info
            else:
                self.adaptive_qs_info = adaptive_query_space_info

        # For adaptive query space
        self.DT = np.abs(self.time_stamps[-1] - self.time_stamps[0])
        self.dt = np.mean(self.time_stamps[1:] - self.time_stamps[0:-1])  # if time step size isn't uniform in the initial range

        # Set the active learner
        self.active_learner = active_learner

    def update_query_space(self, k_growth):
        """
        Method to update query space depending on type of growth and type of oracle (simulator or expt data)

        Note: For simulator, concatenatation ensures old tvec doesn't change and the
        ActionSpace is able to work with old training examples
        """
        growth_time_stamps = self.adaptive_qs_info['growth_time_stamps']
        growth_factor = self.adaptive_qs_info['growth_factor']

        if growth_time_stamps == 'linear':
            t1_mod = self.time_stamps[0] + k_growth * growth_factor * self.DT
        elif growth_time_stamps == 'exponential':
            t1_mod = self.time_stamps[0] + (growth_factor ** (k_growth - 1)) * self.DT
        else:
            raise RuntimeError('Passed invalid growth argument error')

        if self.env_cr.FLAG_simulator:
            t0_mod = self.time_stamps[-1] + self.dt
            time_stamps_mod = np.arange(t0_mod, t1_mod, self.dt)
            self.time_stamps = np.concatenate((self.time_stamps, time_stamps_mod), axis=0)
        else:
            time_stamps_expt_data = self.env_cr.expt_data['query_info']['time']
            queries_interest = np.where(time_stamps_expt_data <= t1_mod + self.dt)[0]
            N_t = int((queries_interest[-1] + 1) / 6)
            self.time_stamps = self.env_cr.expt_data['query_info']['time'][::6][0:N_t]

        self.time_stamps_nd = self.time_stamps / self.xi_t

    def set_query_constraints(self, A_cr, N_tot, N_batch, N_tot_old, q_old):
        if self.FLAG_query_constraints:
            if self.query_optimization_type == 'batch':
                upper_bound_q = [np.amin([A_cr.action_n_shots[ind] / N_batch, 1.0]) for ind in range(A_cr.N_actions)]
                query_constraints = {'N_tot_old': N_tot_old, 'N_tot': N_tot,
                                     'q_old': q_old, 'upper_bound_q': upper_bound_q, 'FLAG_lower_limits': False}
            else:
                query_constraints = {'N_tot_old': N_tot_old, 'N_tot': N_tot,
                                     'q_old': q_old, 'N_shots': self.query_constraints_ref['N_shots'],
                                     'FLAG_lower_limits': True}

            self.active_learner.update(FLAG_constraints=self.FLAG_query_constraints,
                                       query_constraints=query_constraints)

    def query_optimization(self, A_cr, qs_model, N_p, q_vec, p_U):
        # N_tot_old = N_p, N_tot = N_p + N_batch, q_old = q_vec[-1]
        self.set_query_constraints(A_cr, N_p + self.N_batch, self.N_batch, N_p, q_vec[-1])

        # Query optimization
        q_dist = self.active_learner.optimal_query_distribution(A_cr, qs_model,
                                                           p_ref=p_U, FLAG_verbose_solver=False)

        # Mix with uniform distribution over the valid action set
        N_actions_filtered = len(A_cr.filtered_actions())
        p_U_filtered = (1 / N_actions_filtered) * np.ones(N_actions_filtered)
        lambda_m = 1.0 - 1. / ((N_p) ** (1 / 6))
        q_dist = lambda_m * q_dist + (1 - lambda_m) * p_U_filtered

        # Sample from query distribution q noting that some queries have already been made
        X_q = A_cr.sample_action_space(self.env_cr, q_dist, self.N_batch,
                                       actions_list=A_cr.nonempty_actions(), FLAG_query=True)
        return q_dist, X_q

    def hamiltonian_parameter_estimator(self, X_p, A_cr, N_p, init_J=None, verbose=False, do_plot=False):
        """
        Runs the estimator based on "est_param_info"

        Returns:
            loss, parameters

        TODO: Currently not compatible with the full-blown estimation procedure. Look at how init_J_num is being handled
        """
        if self.FLAG_debug_AL:
            init_J_num = self.J_truth
            self.estimation_strategy['FLAG_initial_estimate'] = False
        else:
            # Over-ride given init_J if initial_estimate is required
            if self.estimation_strategy['FLAG_initial_estimate'] is True:
                init_J_num = None
            else:
                init_J_num = init_J

        if self.estimation_strategy['baseline'] is True:
            if verbose:
                print('Using baseline estimator')

            loss_J, J_num = estimation_procedure.baseline_estimate(X_p, A_cr)
        else:
            if self.estimation_strategy['quick_MLE'] is True:
                if verbose:
                    print('Using quick MLE estimator')

                loss_J, J_num = estimation_procedure.quick_mle_estimate(X_p, A_cr, init_J=init_J_num,
                                                                        FLAG_verbose=verbose)
            else:
                if verbose:
                    print('Using a very complicated full-blow MLE solver')

                # Update MLE solver options
                mini_batch_size = np.amin([int(N_p / 5), 512])
                solver_options = {'mini_batch_size': mini_batch_size}

                loss_J, J_num = estimation_procedure.estimation_procedure(X_p, init_J=init_J_num, env_cr=self.env_cr,
                                                                          A_cr=A_cr,
                                                                          solver_options=solver_options,
                                                                          estimation_strategy=self.estimation_strategy,
                                                                          verbose=verbose, do_plot=do_plot)
        return loss_J, J_num

    @staticmethod
    def logger(log_file, k_iter, mse_J, loss_J, J_num):
        """
        Inputs:
            log_file:
            k_iter: Iteration number
            mse_J: MSE in J
            loss_J:
            J_num:

        Returns nothing
        """
        f_log = open(log_file, "a+")
        f_log.write("%d %f %f %f %f %f %f %f %f\n" % (k_iter, np.sqrt(mse_J), loss_J,
                                                      J_num[0], J_num[1], J_num[2],
                                                      J_num[3], J_num[4], J_num[5]))
        f_log.close()

    def logger_experimental_data(self, log_file, mse_train_vec, loss_vec, J_vec, mse_test_vec):
        """
        Logger special to experimental dataset or wherever there is a notion of testing dataset and not ground truth
        Ref: https://stackoverflow.com/questions/2769061/how-to-erase-the-file-contents-of-text-file-in-python

        Inputs:
            log_file:
            k_iter: Iteration number
            mse_J: MSE in J
            loss_J:
            J_num:

        Returns nothing
        """
        f_log = open(log_file, "a")
        f_log.seek(0)
        f_log.truncate()

        for k_iter in range(self.max_iter + 1):
            J_num = np.copy(J_vec[k_iter])
            f_log.write("%d %f %f %f %f %f %f %f %f %f\n" % (k_iter,
                                                             np.sqrt(mse_train_vec[k_iter]),
                                                             loss_vec[k_iter],
                                                             J_num[0], J_num[1], J_num[2],
                                                             J_num[3], J_num[4], J_num[5],
                                                             np.sqrt(mse_test_vec[k_iter])))
        f_log.close()

    def test_query_constraints_pruning(self):
        # Initialize action space
        A_cr = simulate_nature.Action_Space(self.moset, self.prepset, self.time_stamps_nd, self.xi_t,
                                            xi_J=self.xi_J, n_shots=self._max_shots_ActionSpace,
                                            freq_convert=self.freq_convert)

        # Variable definitions
        N_config = A_cr.N_actions  # Number of different configurations/actions in the pool
        loss_num = []
        J_num = []
        J_nd_num = []
        N_p_vec = []
        q_vec = []
        n_shots_vec = []  # Holds the number of shots available in each query

        # Uniform probability distribution over the pool
        p_U = (1 / N_config) * np.ones(N_config)

        # Update number of queries collected so far
        N_p = self.N_0

        # Mixing parameter for active learning procedure (See Chaudhuri et al., 2015; Sourati et al, 2017)
        lambda_m = 1.0 - 1. / ((N_p) ** (1 / 6))

        # Create initial dataset using set_P (and not p_U -- generalize later)
        X_p = A_cr.sample_action_space(self.env_cr, p_U, self.N_0, FLAG_query=True)

        # Update action space with actions sampled
        A_cr.update_dict_action_space(X_p)

        _Jnum = self.J_truth

        J_num.append(_Jnum)
        J_nd_num.append(_Jnum / self.xi_J)
        N_p_vec.append(N_p)
        q_vec.append(p_U)
        n_shots_vec.append(np.copy(A_cr.action_n_shots))

        # Create Quantum System Model based on the estimate so far
        # Noise Model
        qs_noise = {'readout': self.env_cr.readout_noise,
                    'control_noise': functools.partial(quantum_device_models.data_driven_teff_noise_model,
                                                       FLAG_ibmq_boel=True),
                    'grad_control_noise': functools.partial(quantum_device_models.grad_data_driven_teff_noise_model,
                                                            FLAG_ibmq_boel=True)}

        qs_model = quantum_device_models.SystemModel(_Jnum, 1e6 * np.ones(len(_Jnum)), noise=qs_noise,
                                                     FLAG_readout_noise=True, FLAG_control_noise=True)

        print('Optimal query distribution')

        n_iter_growth = 0
        k_growth = 1

        # ActiveLearning + Online optimization
        for k in range(self.max_iter):
            print('Iteration %d' % k)
            # Update counter to check if it's time to grow the query space or not
            n_iter_growth += 1

            if self.FLAG_adaptive_query_space:
                if self.adaptive_qs_info['growth_time_stamps'] is not None and n_iter_growth >= self.adaptive_qs_info['max_iter_growth']:
                    k_growth += 1

                    # Update the time_stamps
                    # Note that the resolution in the old range may not necessarily be the same in the new range
                    # for both the simulator oracle and the experimental data oracle
                    print('Updating query space!')
                    self.update_query_space(k_growth)

                    # Update the action space
                    A_cr = simulate_nature.Action_Space(self.moset, self.prepset, self.time_stamps_nd, self.xi_t,
                                                        xi_J=qs_model.xi_J, xi_param=qs_model.xi_param,
                                                        n_shots=self._max_shots_ActionSpace,
                                                        freq_convert=self.freq_convert)

                    N_config = A_cr.N_actions

                    # Update the dictionary of action space with the samples we already have
                    A_cr.update_dict_action_space(X_p)

                    # Update uniform query distribution as support has now changed
                    p_U = (1 / N_config) * np.ones(N_config)

                    # Reset the counter for growth
                    n_iter_growth = 0

                    # Switch off FLAG_initial_estimate
                    self.FLAG_initial_estimate = False

            # Update the ActiveLearner -- Query Constraints and (soon: adaptively growing query space)
            # N_tot_old = N_p, N_tot = N_p + N_batch, q_old = q_vec[-1]
            self.set_query_constraints(A_cr, N_p + self.N_batch, self.N_batch, N_p, q_vec[-1])

            # Query optimization
            try:
                q = self.active_learner.optimal_query_distribution(A_cr, qs_model,
                                                                   p_ref=p_U, FLAG_verbose_solver=False)
            except:
                print('SDP Solver Failed! Re-running using verbose=True setting')
                # q = self.active_learner.optimal_query_distribution(self.time_stamps_nd, self.xi_t, N_config, qs_model,
                #                                                    p_ref=p_U, FLAG_verbose_solver=True)
                q = self.active_learner.optimal_query_distribution(A_cr, qs_model,
                                                                   p_ref=p_U, FLAG_verbose_solver=True)

            # Mix with uniform distribution over the valid action set
            N_actions_filtered = len(A_cr.filtered_actions())
            p_U_filtered = (1/N_actions_filtered) * np.ones(N_actions_filtered)
            lambda_m = 1.0 - 1. / ((N_p) ** (1 / 6))
            q = lambda_m * q + (1 - lambda_m) * p_U_filtered

            # Sample from query distribution q noting that some queries have already been made
            X_q = A_cr.sample_action_space(self.env_cr, q, self.N_batch,
                                           actions_list=A_cr.nonempty_actions(), FLAG_query=True)

            # Update action space with actions sampled
            A_cr.update_dict_action_space(X_q)

            # Update data with queried dataset -- merge_datasets will probably need to be updated when we have adaptive time_stamps
            X_p = A_cr.merge_datasets(X_p, X_q)
            N_p = N_p + self.N_batch

            # Update variables
            J_num.append(_Jnum)
            J_nd_num.append(_Jnum / self.xi_J)
            N_p_vec.append(N_p)
            q_vec.append(q)
            n_shots_vec.append(np.copy(A_cr.action_n_shots))

        results = {'loss': loss_num, 'J_hat': J_num, 'J_truth': self.J_truth, 'xi_J': self.xi_J,
                   'N_p': N_p_vec, 'q': q_vec, 'n_shots': n_shots_vec, 'data': X_p, 'A_cr': A_cr}

        return results

    def AL_runner(self, verbose=False, do_plot=False, log_file=None):
        """
        Calculates the MLE estimate according to incoming samples
        which are queried through an active learning strategy

        As this is simulated, we just take in values of J, time-stamps and their scalings
        for generating/querying data

        This would be modified when it is a pool of data, etc.

        Inputs:
        env_cr = environment to be used for querying and creating the datasets
        Options to TensorFlow SGD procedure
        query_opt = algorithm for query optimization (default is random)

        Outputs:
        Returns output
        """
        # Initialize action space
        A_cr = simulate_nature.Action_Space(self.moset, self.prepset, self.time_stamps_nd, self.xi_t,
                                            xi_J=self.xi_J, n_shots=self._max_shots_ActionSpace,
                                            freq_convert=self.freq_convert)

        # Variable definitions
        N_config = A_cr.N_actions  # Number of different configurations/actions in the pool
        loss_num = []
        mse_vec = [] # Current error from the truth if given
        J_num = []
        J_nd_num = []
        N_p_vec = []
        q_vec = []
        n_shots_vec = [] # Holds the number of shots available in each query
        N_actions_vec = []  # Holds the number of actions

        # Uniform probability distribution over the pool
        p_U = (1/N_config)*np.ones(N_config)

        n_samples_query_U = round(self.N_0 / N_config)
        set_P = n_samples_query_U * np.ones(N_config)   # Number of shots made so far for each query
        set_Q = np.zeros(N_config)  # Number of shots being considered for each query

        # Update number of queries collected so far
        N_p = self.N_0

        # Mixing parameter for active learning procedure (See Chaudhuri et al., 2015; Sourati et al, 2017)
        lambda_m = 1.0 - 1. / ((N_p) ** (1 / 6))

        # Create initial dataset using set_P (and not p_U -- generalize later)
        X_p = A_cr.sample_action_space(self.env_cr, set_P, self.N_0, FLAG_query=False)

        # Update action space with actions sampled
        A_cr.update_dict_action_space(X_p)

        _loss, _Jnum = self.hamiltonian_parameter_estimator(X_p, A_cr, N_p,
                                                            init_J=self.init_J_num,
                                                            verbose=verbose, do_plot=do_plot)
        loss_num.append(_loss)
        J_num.append(_Jnum)
        J_nd_num.append(_Jnum/self.xi_J)
        N_p_vec.append(N_p)
        q_vec.append(p_U)
        n_shots_vec.append(A_cr.action_n_shots)
        N_actions_vec.append(A_cr.N_actions)

        # Write to log file
        mse_temp = estimation_procedure.normalized_L2_error(_Jnum, self.J_truth, self.xi_J_error) ** 2
        mse_vec.append(mse_temp)

        if log_file is not None:
            self.logger(log_file, 0, mse_temp, _loss, _Jnum)

        # Create Quantum System Model based on the estimate so far
        # Noise Model
        if self.env_cr.FLAG_decoherence:
            qs_T1 = 1e-6 * np.array([94.0278, 75.71162])
            qs_T2 = 1e-6 * np.array([177.22575, 128.0758598])

            qs_noise = {'readout': self.env_cr.readout_noise,
                        'control_noise': functools.partial(quantum_device_models.data_driven_teff_noise_model,
                                                       FLAG_ibmq_boel=True),
                        'grad_control_noise': functools.partial(quantum_device_models.grad_data_driven_teff_noise_model,
                                                            FLAG_ibmq_boel=True),
                        'T1': qs_T1, 'T2': qs_T2,
                        'decoherence': self.env_cr.decoherence_model}

            qs_model = quantum_device_models.SystemModel(_Jnum, self.xi_J_error, noise=qs_noise,
                                                         FLAG_readout_noise=True, FLAG_control_noise=True,
                                                         FLAG_decoherence=True)
        else:
            qs_noise = {'readout': self.env_cr.readout_noise,
                        'control_noise': functools.partial(quantum_device_models.data_driven_teff_noise_model,
                                                       FLAG_ibmq_boel=True),
                        'grad_control_noise': functools.partial(quantum_device_models.grad_data_driven_teff_noise_model,
                                                            FLAG_ibmq_boel=True)}

            qs_model = quantum_device_models.SystemModel(_Jnum, self.xi_J_error, noise=qs_noise,
                                                         FLAG_readout_noise=True, FLAG_control_noise=True)

        n_iter_growth = 0
        k_growth = 1

        # ActiveLearning + Online optimization
        for k in range(self.max_iter):
            # Update counter to check if it's time to grow the query space or not
            n_iter_growth += 1

            if self.FLAG_adaptive_query_space:
                if (self.adaptive_qs_info['growth_time_stamps'] is not None and
                        n_iter_growth >= self.adaptive_qs_info['max_iter_growth']):
                    k_growth += 1

                    # Update the time_stamps
                    # Note that the resolution in the old range may not necessarily be the same in the new range
                    # for both the simulator oracle and the experimental data oracle
                    print('Updating query space!')
                    self.update_query_space(k_growth)

                    # Update the action space
                    A_cr = simulate_nature.Action_Space(self.moset, self.prepset, self.time_stamps_nd, self.xi_t,
                                                        xi_J=qs_model.xi_J, xi_param=qs_model.xi_param,
                                                        n_shots=self._max_shots_ActionSpace,
                                                        freq_convert=self.freq_convert)

                    N_config = A_cr.N_actions

                    print("%d actions in Time range (x 1e-7): (%f,%f)" % (N_config, self.time_stamps_nd[0],
                                                                          self.time_stamps_nd[-1]))

                    # Update the dictionary of action space with the samples we already have
                    n_shots_vec_old = n_shots_vec[k]    # as simply updating won't update the number of shots left
                    N_actions_old = N_actions_vec[k]
                    A_cr.update_dict_action_space(X_p, n_shots_vec_old, N_actions_old)

                    # Update uniform query distribution as support has now changed
                    p_U = (1 / N_config) * np.ones(N_config)

                    # Reset the counter for growth
                    n_iter_growth = 0

                    # Switch off FLAG_initial_estimate
                    self.estimation_strategy['FLAG_initial_estimate'] = False

            # Query Optimization
            q, X_q = self.query_optimization(A_cr, qs_model, N_p, q_vec, p_U)

            # Update action space with actions sampled
            A_cr.update_dict_action_space(X_q)

            # Update data with queried dataset -- merge_datasets needs to be updated for adaptive time_stamps
            X_p = A_cr.merge_datasets(X_p, X_q)
            N_p = N_p + self.N_batch

            # Update Hamiltonian parameters' estimate
            _loss, _Jnum = self.hamiltonian_parameter_estimator(X_p, A_cr, N_p,
                                                                init_J=J_num[k],
                                                                verbose=verbose, do_plot=do_plot)

            # Update SystemModel
            qs_model.update(_Jnum)

            # Update variables
            loss_num.append(_loss)
            J_num.append(_Jnum)
            J_nd_num.append(_Jnum / self.xi_J)
            N_p_vec.append(N_p)
            q_vec.append(q)
            n_shots_vec.append(A_cr.action_n_shots)
            N_actions_vec.append(A_cr.N_actions)

            mse_temp = estimation_procedure.normalized_L2_error(_Jnum, self.J_truth, self.xi_J_error) ** 2
            mse_vec.append(mse_temp)

            # Write to log file
            if log_file is not None:
                self.logger(log_file, k+1, mse_temp, _loss, _Jnum)

        # Results so far
        results = {'loss': loss_num, 'mse': mse_vec, 'J_hat': J_num, 'J_truth': self.J_truth, 'xi_J': self.xi_J,
                   'N_p': N_p_vec, 'q': q_vec, 'n_shots': n_shots_vec, 'data': X_p, 'A_cr': A_cr}

        if self.env_cr.FLAG_simulator is False:
            # Don't do this for growing query space as simulate_nature.get_testing_dataset can't handle this yet
            if not self.FLAG_adaptive_query_space:
                # Get RMSE with respect to a testing dataset as well
                # Create testing dataset and associated ActionSpace
                X_test, A_cr_test = simulate_nature.get_testing_dataset(self.env_cr, A_cr)

                # Get "estimate" of J from the testing dataset using J_truth as guess
                # We include J_truth as n_shots left over may be 0 for certain queries and hence rabi isn't reliable
                _, J_num_test = estimation_procedure.quick_mle_estimate(X_test, A_cr_test, init_J=self.J_truth)

                mse_test = np.zeros(self.max_iter + 1)

                # Compute the testing RMSE
                for ind in range(self.max_iter + 1):
                    mse_test[ind] = estimation_procedure.normalized_L2_error(J_num[ind], J_num_test,
                                                                             self.xi_J_error) ** 2

                if log_file is not None:
                    self.logger_experimental_data(log_file, mse_vec, loss_num, J_num, mse_test)

                # Update results -- not including testing dataset for the time-being
                results.update({'mse_test': mse_test})
                results.update({'J_test': J_num_test})

        return results

    def PL_runner(self, verbose=False, do_plot=False, log_file=None):
        """
        Calculates the MLE estimate according to incoming samples
        which are queried through an active learning strategy

        As this is simulated, we just take in values of J, time-stamps and their scalings
        for generating/querying data

        This would be modified when it is a pool of data, etc.

        Inputs:
        env_cr = environment to be used for querying and creating the datasets
        Options to TensorFlow SGD procedure
        query_opt = algorithm for query optimization (default is random)

        Outputs:
        Returns output
        """

        # Initialize action space
        A_cr = simulate_nature.Action_Space(self.moset, self.prepset, self.time_stamps_nd, self.xi_t,
                                            xi_J=self.xi_J, n_shots=self._max_shots_ActionSpace,
                                            freq_convert=self.freq_convert)

        # Variable definitions
        N_config = A_cr.N_actions  # Number of different configurations/actions in the pool
        loss_num = []
        mse_vec = [] # Current error from the truth if given
        J_num = []
        J_nd_num = []
        N_p_vec = []
        q_vec = []
        n_shots_vec = [] # Holds the number of shots available in each query

        # Uniform probability distribution over the pool
        p_U = (1/N_config)*np.ones(N_config)

        n_samples_query_U = round(self.N_0 / N_config)
        set_P = n_samples_query_U * np.ones(N_config)   # Number of shots made so far for each query

        # Update number of queries collected so far
        N_p = self.N_0

        # Create initial dataset using set_P (and not p_U -- generalize later)
        X_p = A_cr.sample_action_space(self.env_cr, set_P, self.N_0, FLAG_query=False)

        # Update action space with actions sampled
        A_cr.update_dict_action_space(X_p)

        _loss, _Jnum = self.hamiltonian_parameter_estimator(X_p, A_cr, N_p,
                                                            init_J=self.init_J_num,
                                                            verbose=verbose, do_plot=do_plot)

        loss_num.append(_loss)
        J_num.append(_Jnum)
        J_nd_num.append(_Jnum/self.xi_J)
        N_p_vec.append(N_p)
        q_vec.append(p_U)
        n_shots_vec.append(A_cr.action_n_shots)

        # Write to log file
        mse_temp = estimation_procedure.normalized_L2_error(_Jnum, self.J_truth, self.xi_J_error) ** 2
        mse_vec.append(mse_temp)

        if log_file is not None:
            self.logger(log_file, 0, mse_temp, _loss, _Jnum)

        # Create Quantum System Model based on the estimate so far
        # Noise Model
        qs_noise = {'readout': self.env_cr.readout_noise,
                    'control_noise': functools.partial(quantum_device_models.data_driven_teff_noise_model,
                                                       FLAG_ibmq_boel=True),
                    'grad_control_noise': functools.partial(quantum_device_models.grad_data_driven_teff_noise_model,
                                                            FLAG_ibmq_boel=True)}

        qs_model = quantum_device_models.SystemModel(_Jnum, 1e6 * np.ones(len(_Jnum)), noise=qs_noise,
                                                     FLAG_readout_noise=True, FLAG_control_noise=True)

        print('Passive Learning -- Uniform query distribution')

        # Passive Learning
        for k in range(self.max_iter):
            # Update the query constraints
            # N_tot_old = N_p, N_tot = N_p + N_batch, q_old = q_vec[-1]
            self.set_query_constraints(A_cr, N_p + self.N_batch, self.N_batch, N_p, q_vec[-1])

            # Sample from uniform distribution p_U
            n_samples_query_U = round(self.N_batch / N_config)
            set_Q = n_samples_query_U * np.ones(N_config)  # Number of shots made so far for each query
            X_q = A_cr.sample_action_space(self.env_cr, set_Q, self.N_batch, FLAG_query=False)

            # Update action space with actions sampled
            A_cr.update_dict_action_space(X_q)

            # Update data with queried dataset -- merge_datasets
            X_p = A_cr.merge_datasets(X_p, X_q)
            N_p = N_p + self.N_batch

            # Update Hamiltonian parameters' estimate
            _loss, _Jnum = self.hamiltonian_parameter_estimator(X_p, A_cr, N_p,
                                                                init_J=J_num[k],
                                                                verbose=verbose, do_plot=do_plot)

            # Update SystemModel
            qs_model.update(_Jnum)

            # Update variables
            loss_num.append(_loss)
            J_num.append(_Jnum)
            J_nd_num.append(_Jnum / self.xi_J)
            N_p_vec.append(N_p)
            q_vec.append(p_U)
            n_shots_vec.append(A_cr.action_n_shots)

            mse_temp = estimation_procedure.normalized_L2_error(_Jnum, self.J_truth, self.xi_J_error) ** 2
            mse_vec.append(mse_temp)

            # Write to log file
            if log_file is not None:
                self.logger(log_file, k+1, mse_temp, _loss, _Jnum)

        # Results so far
        results = {'loss': loss_num, 'mse': mse_vec, 'J_hat': J_num, 'J_truth': self.J_truth, 'xi_J': self.xi_J,
                   'N_p': N_p_vec, 'q': q_vec, 'n_shots': n_shots_vec, 'data': X_p, 'A_cr': A_cr}

        if self.env_cr.FLAG_simulator is False:
            # Get RMSE with respect to a testing dataset as well
            # Create testing dataset and associated ActionSpace
            X_test, A_cr_test = simulate_nature.get_testing_dataset(self.env_cr, A_cr)

            # Get "estimate" of J from the testing dataset
            _, J_num_test = estimation_procedure.quick_mle_estimate(X_test, A_cr_test)

            mse_test = np.zeros(self.max_iter + 1)

            # Compute the testing RMSE
            for ind in range(self.max_iter + 1):
                mse_test[ind] = estimation_procedure.normalized_L2_error(J_num[ind], J_num_test, self.xi_J_error) ** 2

            if log_file is not None:
                self.logger_experimental_data(log_file, mse_vec, loss_num, J_num, mse_test)

            # Update results -- not including testing dataset for the time-being
            results.update({'mse_test': mse_test})
            results.update({'J_test': J_num_test})

        return results


# The definitive function to plot rabi oscillations and for all needs
def plot_rabi_oscillations(time_stamps, pvec_data=None, qs_model=None, FLAG_noise=True,
                           figsize_plot=(10,7), FLAG_save=False, save_filename='llong_rabi_oscillations.png'):
    """
    This function plots the data with a lot of processing as required

    :param J_num:
    :param data:
    :param FLAG_readout_noise:
    :return:
    """
    # Calculate model probabilities
    if not (qs_model is None):
        # Get a more dense array of time_stamps
        DT = time_stamps[-1] - time_stamps[0]
        n_t = int(DT/(time_stamps[1] - time_stamps[0]))
        time_stamps_model = np.linspace(time_stamps[0], time_stamps[-1], n_t)

        pvec_model = np.zeros((6,len(time_stamps_model)))

        for n_config in range(6):
            pvec_model[n_config, :] = qs_model.plot_rabi_oscillations(n_config, time_stamps_model, FLAG_noise=FLAG_noise)

    # To force xlabels and ylabels to use the right defined Font
    # Ref: https://stackoverflow.com/questions/11611374/sans-serif-font-for-axes-tick-labels-with-latex
    from matplotlib import rc
    rc('text.latex', preamble=r'\usepackage{sfmath}')

    fig = plt.figure(3, figsize=figsize_plot)
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['text.usetex'] = True

    # plt.rcParams['figure.constrained_layout.use'] = True

    plt.subplot(311)
    plt.tick_params(axis='both', direction='in', labelbottom=False)
    if not (qs_model is None):
        plt.plot(time_stamps_model, pvec_model[0, :], '-b', linewidth=1.5, label='model')
        plt.plot(time_stamps_model, pvec_model[1, :], '-r', linewidth=1.5)

    if not (pvec_data is None):
        plt.plot(time_stamps, pvec_data[0, :], 'ob', markersize=5, label='data')
        plt.plot(time_stamps, pvec_data[1, :], '^r', markersize=5)

    plt.axhline(y=1, linestyle='--', color='k')
    plt.axhline(y=-1, linestyle='--', color='k')
    plt.ylabel(r"$\langle X \rangle$")
    plt.ylim((-1.2, 1.2))
    # plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.yticks(np.arange(-1, 1.5, step=0.5))
    # plt.gca().set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])

    plt.subplot(312)
    plt.tick_params(axis='both', direction='in', labelbottom=False)
    if not (qs_model is None):
        plt.plot(time_stamps_model, pvec_model[2, :], '-b', linewidth=1.5, label='model')
        plt.plot(time_stamps_model, pvec_model[3, :], '-r', linewidth=1.5)

    if not (pvec_data is None):
        plt.plot(time_stamps, pvec_data[2, :], 'ob', markersize=5, label='data')
        plt.plot(time_stamps, pvec_data[3, :], '^r', markersize=5)

    plt.axhline(y=1, linestyle='--', color='k')
    plt.axhline(y=-1, linestyle='--', color='k')
    plt.ylabel(r"$\langle Y \rangle$")
    plt.ylim((-1.2, 1.2))
    plt.yticks(np.arange(-1, 1.5, step=0.5))

    plt.subplot(313)
    plt.tick_params(axis='both', direction='in')
    if not (qs_model is None):
        L1 = plt.plot(time_stamps_model, pvec_model[4, :], '-b', linewidth=1.5, label='model')
        L2 = plt.plot(time_stamps_model, pvec_model[5, :], '-r', linewidth=1.5)

    if not (pvec_data is None):
        L3 = plt.plot(time_stamps, pvec_data[4, :], 'ob', markersize=5, label='data')
        L4 = plt.plot(time_stamps, pvec_data[5, :], '^r', markersize=5)

    plt.axhline(y=1, linestyle='--', color='k')
    plt.axhline(y=-1, linestyle='--', color='k')
    plt.xlabel("t (s)")
    plt.ylabel(r"$\langle Z \rangle$")
    plt.ylim((-1.2, 1.2))
    plt.yticks(np.arange(-1, 1.5, step=0.5))

    # Legend properties
    if (qs_model is not None) and (pvec_data is not None):
        # Refs:
        # 1. https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib
        # 2. Why using [0] for the line objects? https://discourse.julialang.org/t/userwarning-legend-does-not-support-line2d-object/970/3
        labels_fig = (r'Model (Control in $|0\rangle$)', r'Model (Control in $|1\rangle$)',
                      r'Data (Control in $|0\rangle$)', r'Data (Control in $|1\rangle$)')

        fig.legend((L1[0], L2[0], L3[0], L4[0]), labels_fig, bbox_to_anchor=(0., -0.125, 1., 0.02),
                   loc='lower center', ncol=2, labelspacing=0., borderaxespad=0., fontsize=22)
    elif (pvec_data is not None) and (qs_model is None):
        labels_fig = (r'Control in $|0\rangle$', r'Control in $|1\rangle$')

        fig.legend((L3[0], L4[0]), labels_fig, bbox_to_anchor=(0., -0.125, 1., 0.02),
                   loc='lower center', ncol=2, labelspacing=0., fontsize=22)
    elif (pvec_data is None) and (qs_model is not None):
        labels_fig = (r'Control in $|0\rangle$', r'Control in $|1\rangle$')

        fig.legend((L1[0], L2[0]), labels_fig, bbox_to_anchor=(0., -0.125, 1., 0.02),
                   loc='lower center', ncol=2, labelspacing=0., fontsize=22)

    plt.tight_layout()
    if FLAG_save is True:
        plt.savefig(save_filename, bbox_inches='tight', dpi=600)

    plt.show()

    if qs_model is not None:
        return pvec_model


def compare_rabi_oscillations(time_stamps, qs_model_sim, qs_model_expt, pvec_data=None, FLAG_noise=True,
                              figsize_plot=(10,7), FLAG_save=False, save_filename='llong_rabi_oscillations.png'):
    """
    Compares the difference rabi oscillations produced by two different models

    :param J_num:
    :param data:
    :param FLAG_readout_noise:
    :return:
    """
    # Calculate model probabilities
    # Get a more dense array of time_stamps
    DT = time_stamps[-1] - time_stamps[0]
    n_t = int(DT/(time_stamps[1] - time_stamps[0]))
    time_stamps_model = np.linspace(time_stamps[0], time_stamps[-1], n_t)

    pvec_model_sim = np.zeros((6,len(time_stamps_model)))
    pvec_model_expt = np.zeros((6, len(time_stamps_model)))

    for n_config in range(6):
        pvec_model_sim[n_config, :] = qs_model_sim.plot_rabi_oscillations(n_config, time_stamps_model, FLAG_noise=FLAG_noise)
        pvec_model_expt[n_config, :] = qs_model_expt.plot_rabi_oscillations(n_config, time_stamps_model, FLAG_noise=FLAG_noise)

    pvec_model_diff = pvec_model_expt - pvec_model_sim

    fig = plt.figure(3, figsize=figsize_plot)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24

    # plt.rcParams['figure.constrained_layout.use'] = True

    plt.subplot(311)
    plt.tick_params(axis='both', direction='in', labelbottom=False)

    plt.plot(time_stamps_model, pvec_model_diff[0, :], '-b', label='model')
    plt.plot(time_stamps_model, pvec_model_diff[1, :], '-r')

    if not (pvec_data is None):
        plt.plot(time_stamps, pvec_data[0, :], 'ob', label='data')
        plt.plot(time_stamps, pvec_data[1, :], '^r')

    plt.axhline(y=1, linestyle='--', color='k')
    plt.axhline(y=-1, linestyle='--', color='k')
    plt.ylabel(r"$\langle X \rangle$")
    plt.ylim((-1.2, 1.2))
    # plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.yticks(np.arange(-1, 1.5, step=0.5))
    # plt.gca().set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])

    plt.subplot(312)
    plt.tick_params(axis='both', direction='in', labelbottom=False)

    plt.plot(time_stamps_model, pvec_model_diff[2, :], '-b', label='model')
    plt.plot(time_stamps_model, pvec_model_diff[3, :], '-r')

    if not (pvec_data is None):
        plt.plot(time_stamps, pvec_data[2, :], 'ob', label='data')
        plt.plot(time_stamps, pvec_data[3, :], '^r')

    plt.axhline(y=1, linestyle='--', color='k')
    plt.axhline(y=-1, linestyle='--', color='k')
    plt.ylabel(r"$\langle Y \rangle$")
    plt.ylim((-1.2, 1.2))
    plt.yticks(np.arange(-1, 1.5, step=0.5))

    plt.subplot(313)
    plt.tick_params(axis='both', direction='in')

    L1 = plt.plot(time_stamps_model, pvec_model_diff[4, :], '-b', label='model')
    L2 = plt.plot(time_stamps_model, pvec_model_diff[5, :], '-r')

    if not (pvec_data is None):
        L3 = plt.plot(time_stamps, pvec_data[4, :], 'ob', label='data')
        L4 = plt.plot(time_stamps, pvec_data[5, :], '^r')

    plt.axhline(y=1, linestyle='--', color='k')
    plt.axhline(y=-1, linestyle='--', color='k')
    plt.xlabel("t (s)")
    plt.ylabel(r"$\langle Z \rangle$")
    plt.ylim((-1.2, 1.2))
    plt.yticks(np.arange(-1, 1.5, step=0.5))

    # Legend properties
    if pvec_data is not None:
        # Refs:
        # 1. https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib
        # 2. Why using [0] for the line objects? https://discourse.julialang.org/t/userwarning-legend-does-not-support-line2d-object/970/3
        labels_fig = (r'Model (Control in $|0\rangle$)', r'Model (Control in $|1\rangle$)',
                      r'Data (Control in $|0\rangle$)', r'Data (Control in $|1\rangle$)')

        fig.legend((L1[0], L2[0], L3[0], L4[0]), labels_fig, bbox_to_anchor=(0., -0.1, 1., 0.02),
                   loc='lower center', ncol=2, labelspacing=0., borderaxespad=0., fontsize=22)

    else:
        labels_fig = (r'Control in $|0\rangle$', r'Control in $|1\rangle$')

        fig.legend((L1[0], L2[0]), labels_fig, bbox_to_anchor=(0., -0.1, 1., 0.02),
                   loc='lower center', ncol=2, labelspacing=0., fontsize=22)

    plt.tight_layout()
    if FLAG_save is True:
        plt.savefig(save_filename, bbox_inches='tight', dpi=600)

    plt.show()
