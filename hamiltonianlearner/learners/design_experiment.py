import numpy as np
import scipy.linalg
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
#import mosek
import cvxopt
import cvxpy as cp

import time
from ..quantum_system_oracles import simulate_nature


class ActiveLearner(object):
    """
    Represent the ActiveLearner's knowledge of the oracle (Nature)/environment
    i.e. SystemModel and how it uses this to select queries from ActionSpace
    """

    def __init__(self, policy_type='FIR', FLAG_normalization=False,
                 FLAG_noise=False, type_param='param', FLAG_constraints=False, query_constraints=None):
        """
        policy_type = 'passive' (passive learner), 'FI' (fisher information), 'FIR' (fisher information ratio)

        Required keys in query_constraints
        N_tot_old = N_tot from the previous iteration
        N_tot = For the current iteration
        q_old = query distribution used from the previous iteration (e.g., in the first iteration, it will be q=p_U)
        N_shots = number of overall shots available for x (Assuming this to be same for all x for the time being)
        """

        self.policy_type = policy_type
        self.FLAG_normalization = FLAG_normalization

        # Does the Learner want to consider the presence of noise irrespective of what is there is in the SystemModel
        self.FLAG_noise = FLAG_noise

        # Which parameterization does the Learner want to focus on?
        self.type_param = type_param

        # Including constraints when solving for the optimal query distribution -- right now only shots limitation
        self.FLAG_constraints = FLAG_constraints

        if FLAG_constraints:
            self.query_constraints = query_constraints
        else:
            self.query_constraints = None

    def update(self, FLAG_constraints, query_constraints=None):
        # To allow for updating at a later time ... For the other things, a new instance of the object has to be made

        self.FLAG_constraints = FLAG_constraints

        if FLAG_constraints:
            self.query_constraints = query_constraints
        else:
            self.query_constraints = None

    # Function to calculate total Fisher information given a querying probability distribution
    def fisher_information_query(self, time_stamps, xi_t, qs_model, query_pdf):
        """

        :param time_stamps (normalized)
        :param xi_t (normalization factor)
        :param query_pdf
        :return:
        """
        fisher_info = np.zeros((6, 6))
        counter = 0
        for ind_t in range(len(time_stamps)):
            t_nd = time_stamps[ind_t]
            t = t_nd * xi_t

            for n_config in range(6):
                fisher_info = fisher_info + query_pdf[counter] * qs_model.fisher_information(n_config, t,
                                                                             FLAG_noise=self.FLAG_noise,
                                                             FLAG_normalization=self.FLAG_normalization,
                                                                         type_param=self.type_param)

                counter += 1

        return fisher_info

    def fisher_information_query_action_space(self, action_space, xi_t, qs_model, query_pdf):
        """

        :param time_stamps (normalized)
        :param xi_t (normalization factor)
        :param query_pdf
        :return:
        """
        fisher_info = np.zeros((6, 6))

        n_actions = len(action_space)
        for ind_action in range(n_actions):
            action_temp = action_space[ind_action]
            t_nd = action_temp[2]
            t = t_nd * xi_t

            n_config = 2*action_temp[0] + action_temp[1]

            fisher_info = fisher_info + query_pdf[ind_action] * qs_model.fisher_information(n_config, t,
                                                                         FLAG_noise=self.FLAG_noise,
                                                         FLAG_normalization=self.FLAG_normalization,
                                                                     type_param=self.type_param)

        return fisher_info

    # Function to calculate total Fisher information given a querying probability distribution
    def fisher_information_query_MF(self, time_stamps, xi_t, qs_model, query_pdf):
        """
        A lot of hard-coding at the moment ... Need to rewrite to generalize

        :param time_stamps:
        :param param_nd:
        :param xi_param:
        :param xi_t:
        :param query_pdf:
        :return:
        """

        fisher_info = np.zeros((6, 6))
        counter = 0
        for ind_t in range(len(time_stamps)):
            t_nd = time_stamps[ind_t]
            t = t_nd * xi_t

            for n_config in range(6):
                fisher_info = fisher_info + (query_pdf[counter] / 6) * qs_model.fisher_information(n_config, t,
                                                                                                   FLAG_noise=self.FLAG_noise,
                                                                                                   FLAG_normalization=self.FLAG_normalization,
                                                                                                   type_param=self.type_param)

            counter += 1

        return fisher_info

    def optimal_query_distribution(self, A_cr, qs_model, p_ref=None, threshold_query=1e-5,
                                   FLAG_threshold = True, solver_option=None, FLAG_verbose_solver=False):
        """
        MAKE SURE THE CONFIGURATIONS CORRECTLY CORRESPOND TO THE INDEX
        I.E., CHECK HOW VALUES OF M AND U RELATE TO CONFIG INDEX

        Inputs:
        param_nd = Current running values of the estimate of the param_array (non-dim)
        time_stamps = Time-stamps we can query (non-dim)
        xi_t = non-dimensionalization factors for time
        xi_param = non-dimensionalization factors for components of param_array
        p_ref = testing distribution (Chaudhuri's algo assumed this to be uniform)
        threshold_query = used for removing small values in the distribution
        FLAG_FIR = True indicates use of testing distrn and False just minimizes tr(F^-1)

        FLAG_noise = False indicates using noiseless matrix and True indicates using noisy version
        env_qs = Nature object (used only if FLAG_noise is set to True)

        :return: optimal query distribution

        Updates to code should include this making more general with an input of the pool
        and not making an assumption about what the pool is
        Perhaps should have a class defining the pool and the methods associated with it

        Should also allow for the addition of additional constraints such as the total
        number of experiments that you can do for any config within the pool
        """

        # Define relevant parameters
        valid_action_set = A_cr.nonempty_actions() # Needs to be updated when we give more options such as uncertainty filtering ... right now only non-empty ... Holds indices of actions
        action_list = A_cr.filtered_actions()
        N_config = len(action_list)

        if self.policy_type == 'FIR':
            # Calculate total Fisher Information considering testing distribution (e.g., uniform in Chaudhuri's)
            F_unif = self.fisher_information_query(A_cr.tset, A_cr.xi_t, qs_model, p_ref)
        elif self.policy_type == 'FI':
            # Sourati (2018) algorithm is the same as Chaudhuri (2015) algorithm by setting the testing FI to be identity
            F_unif = np.eye(len(qs_model.J))
        elif self.policy_type == 'passive':
            return np.ones(N_config) / N_config
        elif self.policy_type == 'max_entropy':
            q_entropy, _ = qs_model.filter_actions_entropy(A_cr.action_space, A_cr.xi_t*A_cr.tset, xi_t=A_cr.xi_t,
                                                           FLAG_noise=self.FLAG_noise, threshold=0.99,
                                                           FLAG_filter=False)
            return q_entropy/np.sum(q_entropy)
        else:
            print('Unknown policy type given!')
            return None

        U_unif, S_unif, V_unif = np.linalg.svd(F_unif)

        u_unif = []
        sigma_unif = []

        for ind in range(6):
            u_unif.append(U_unif[:, ind])
            sigma_unif.append(S_unif[ind])

        # To hold constraints associated with the SDP Matrices
        A = []

        # Define and solve the convex optimization problem
        q = cp.Variable(N_config)   # query distribution
        t = cp.Variable(6)          # auxillary variables

        for i in range(6):
            Ai_1 = cp.bmat([[t[i], cp.hstack(u_unif[i])]])
            Ai_2 = cp.bmat(
                [[cp.vstack(u_unif[i]), self.fisher_information_query_action_space(action_list, A_cr.xi_t, qs_model, q)]])

            A.append(cp.vstack([Ai_1, Ai_2]))

        #import pdb; pdb.set_trace()
        # Note: The operator >> denotes matrix inequality.

        # Set constraints -- Maybe divide into common and specific later
        if self.FLAG_constraints:
            print('Solving with query constraints')
            if self.query_constraints['FLAG_lower_limits']:
                print('Lower limits being set')

                N_tot_old = self.query_constraints['N_tot_old']
                N_tot = self.query_constraints['N_tot']
                q_old = self.query_constraints['q_old']
                N_shots = self.query_constraints['N_shots']

                upper_bound_q = np.amin([1,N_shots/N_tot])
                lower_bound_q = (N_tot_old/N_tot)*q_old

                constraints = [cp.sum(q) == 1]
                constraints += [lower_bound_q <= q]
                constraints += [q <= upper_bound_q]
                constraints += [
                    A[i] >> 0 for i in range(6)
                ]
            else:
                print('Upper limits being set')

                _ub_q_temp = self.query_constraints['upper_bound_q']
                upper_bound_q = [_ub_q_temp[ind] for ind in valid_action_set]

                constraints = [cp.sum(q) == 1]
                constraints += [0 <= q]
                constraints += [q <= upper_bound_q]
                constraints += [
                    A[i] >> 0 for i in range(6)
                ]
        else:
            constraints = [cp.sum(q) == 1]
            constraints += [0 <= q]
            constraints += [q <= 1]
            constraints += [
                A[i] >> 0 for i in range(6)
            ]

        # if FLAG_verbose_solver:
        #     import pdb; pdb.set_trace()

        # Solve the SDP Problem
        prob = cp.Problem(cp.Minimize(cp.sum( cp.multiply(sigma_unif,t) )), constraints)

        # if solver_option is 'mosek':
        #     print('Using MOSEK')
        #     prob.solve(solver=cp.MOSEK)
        # else:
        #print('Using DEFAULT')
        #prob.solve(solver=cp.CVXOPT)
        prob.solve(solver=cp.MOSEK, verbose=FLAG_verbose_solver)

        p_query = np.array(q.value)

        # Should add a check if the constraints are satisfied within some tolerance

        if FLAG_threshold is True:
            # Remove negative values
            #import pdb; pdb.set_trace()
            p_query[p_query < 0] = 0
            # Threshold to remove negligible values (may update to introducing L1 constraint)
            p_query[np.abs(p_query) <= threshold_query] = 0
            # Renormalize
            p_query = p_query / np.sum(p_query)

        return p_query

    def optimal_query_distribution_MF(self, time_stamps, xi_t, N_actions_t, qs_model, p_ref=None, threshold_query=1e-5,
                                   FLAG_threshold = True, solver_option=None):
        """
        Active learning assuming a mean-field approximation

        (Need to generalize the active learning code to take in more general structures)

        MAKE SURE THE CONFIGURATIONS CORRECTLY CORRESPOND TO THE INDEX
        I.E., CHECK HOW VALUES OF M AND U RELATE TO CONFIG INDEX

        Inputs:
        param_nd = Current running values of the estimate of the param_array (non-dim)
        time_stamps = Time-stamps we can query (non-dim)
        xi_t = non-dimensionalization factors for time
        xi_param = non-dimensionalization factors for components of param_array
        p_ref = testing distribution (Chaudhuri's algo assumed this to be uniform)
        threshold_query = used for removing small values in the distribution
        FLAG_FIR = True indicates use of testing distrn and False just minimizes tr(F^-1)

        :return: optimal query distribution

        Updates to code should include this making more general with an input of the pool
        and not making an assumption about what the pool is
        Perhaps should have a class defining the pool and the methods associated with it

        Should also allow for the addition of additional constraints such as the total
        number of experiments that you can do for any config within the pool

        """

        if self.policy_type == 'FIR':
            # Calculate total Fisher Information considering testing distribution (e.g., uniform in Chaudhuri's)
            F_unif = self.fisher_information_query(time_stamps, xi_t, qs_model, p_ref)
        elif self.policy_type == 'FI':
            # Sourati (2018) is the same as Chaudhuri (2015) algorithm by setting the testing FI to be identity
            F_unif = np.eye(len(qs_model.J))
        elif self.policy_type == 'passive':
            return np.ones(6*N_actions_t) / (6*N_actions_t)
        else:
            print('Unknown policy type given!')
            return None

        U_unif, S_unif, V_unif = np.linalg.svd(F_unif)

        u_unif = []
        sigma_unif = []

        for ind in range(6):
            u_unif.append(U_unif[:, ind])
            sigma_unif.append(S_unif[ind])

        # To hold constraints associated with the SDP Matrices
        A = []

        # Define and solve the convex optimization problem
        q_t = cp.Variable(N_actions_t)   # marginal query distribution over time t
        t = cp.Variable(6)          # auxillary variables

        for i in range(6):
            Ai_1 = cp.bmat([[t[i], cp.hstack(u_unif[i])]])
            Ai_2 = cp.bmat(
                [[cp.vstack(u_unif[i]), self.fisher_information_query_MF(time_stamps, xi_t, qs_model, q_t)]])
            A.append(cp.vstack([Ai_1, Ai_2]))

        # The operator >> denotes matrix inequality.
        # Set constraints -- Maybe divide into common and specific later
        if self.FLAG_constraints:
            print('Solving with query constraints')
            N_tot_old = self.query_constraints['N_tot_old']
            N_tot = self.query_constraints['N_tot']
            q_old = self.query_constraints['q_old']
            N_shots = self.query_constraints['N_shots']

            upper_bound_q = np.amin([1, N_shots / N_tot])
            lower_bound_q = (N_tot_old / N_tot) * q_old

            constraints = [cp.sum(q_t) == 1]
            constraints += [lower_bound_q <= q_t]
            constraints += [q_t <= upper_bound_q]
            constraints += [
                A[i] >> 0 for i in range(6)
            ]
        else:
            constraints = [cp.sum(q_t) == 1]
            constraints += [0 <= q_t]
            constraints += [q_t <= 1]
            constraints += [
                A[i] >> 0 for i in range(6)
            ]

        # Solve the SDP Problem
        prob = cp.Problem(cp.Minimize(cp.sum( cp.multiply(sigma_unif,t) )), constraints)
        prob.solve(solver=cp.MOSEK)

        q = np.array(q_t.value)
        q = np.tile(q,[6,1])
        q = np.ndarray.flatten(q, order='F')
        q = (1/6)*q

        p_query = np.copy(q)

        # Remove negative values
        p_query[p_query < 0] = 0
        # Threshold to remove negligible values (may update to introducing L1 constraint)
        p_query[np.abs(p_query) <= threshold_query] = 0
        # Renormalize
        p_query = p_query / np.sum(p_query)

        return p_query


def optimal_scalings(qs_model, query_space, xi_t, n_batches, max_k, HAL_FI_param,
                     env_qs=None,
                     FLAG_lower_limits=False,
                     freq_convert=None, growth_time_stamps=None, growth_factor=2,
                     FLAG_time_profile=False, FLAG_verbose=False):
    """
    Function to calculate the optimal scalings given the parameters param_nd

    Setup SystemModel in the asymptotic regime
    Compute the optimal query distribution
    """
    print('Running with lower_limits on query constraints: %s' % FLAG_lower_limits)

    # Fetch the elements of the query space
    moset = query_space['moset']
    prepset = query_space['prepset']
    time_stamps = query_space['time_stamps']

    # Create some data structures
    time_stamps_nd = time_stamps / xi_t

    # Define arrays for RMSE and Testing Error
    RMSE_param_array = np.zeros(max_k)
    TE_param_array = np.zeros(max_k)

    if FLAG_time_profile is True:
        time_profile = np.zeros(max_k-1)

    # Running Fisher Information matrix
    FI_param_running = np.zeros(shape=(6, 6))

    # Initialize action space
    A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps_nd, xi_t,
                                        xi_J=qs_model.xi_J, xi_param=qs_model.xi_param, freq_convert=freq_convert)

    # Variable definitions
    N_config = A_cr.N_actions  # Number of different configurations/actions in the pool
    N_p_vec = []
    q_J_vec = []
    q_param_vec = []

    # Uniform probability distribution over the pool
    p_U = (1 / N_config) * np.ones(N_config)

    # Act like you're sampling from the action space
    if env_qs is not None:
        X_p = A_cr.sample_action_space(env_qs, p_U, n_batches[0])

    # Compute current Fisher-Information Matrix
    FI_param_temp = HAL_FI_param.fisher_information_query(time_stamps_nd, xi_t, qs_model, p_U)
    FI_unif = HAL_FI_param.fisher_information_query(time_stamps_nd, xi_t, qs_model, p_U)

    # Update running Fisher-Information Matrices
    FI_param_running = FI_param_running + n_batches[0] * FI_param_temp

    # Update arrays of RMSE and Testing Error
    RMSE_param_array[0] = np.sqrt(np.trace(np.linalg.inv(FI_param_running)))
    TE_param_array[0] = np.trace(np.linalg.inv(FI_param_running)@FI_unif)

    # Update other arrays
    N_p = n_batches[0]
    N_p_vec.append(N_p)
    q_param_vec.append(p_U)

    # For adaptive query space
    DT = np.abs(time_stamps[-1] - time_stamps[0])
    dt = np.abs(time_stamps[1] - time_stamps[0])

    # ActiveLearning Run
    for k in range(1, max_k):
        if growth_time_stamps is not None:
            # Update the time_stamps
            if growth_time_stamps == 'linear':
                time_stamps = np.arange(time_stamps[0], time_stamps[0] + growth_factor*DT, dt)

            elif growth_time_stamps == 'exponential':
                time_stamps = np.arange(time_stamps[0], time_stamps[0] + (growth_factor**(k-1))*DT, dt)

            time_stamps_nd = time_stamps / xi_t

            # Update the action space
            A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps, xi_t,
                                                xi_J=qs_model.xi_J, xi_param=qs_model.xi_param,
                                                freq_convert=freq_convert)

            N_config = A_cr.N_actions

            # Update uniform query distribution as support has now changed
            p_U = (1 / N_config) * np.ones(N_config)

        # Update the ActiveLearner -- Query Constraints and (soon: adaptively growing query space)
        if HAL_FI_param.FLAG_constraints:
            N_tot = N_p + n_batches[k]
            if FLAG_lower_limits:
                query_constraints = {'N_tot_old': N_p, 'N_tot': N_tot,
                                     'q_old': q_param_vec[-1], 'N_shots': HAL_FI_param.query_constraints,
                                     'FLAG_lower_limits': True}
            else:
                upper_bound_q = [np.amin([A_cr.action_n_shots[ind]/A_cr.max_n_shots,1]) for ind in range(A_cr.N_actions)]
                query_constraints = {'N_tot_old': N_p, 'N_tot': N_tot,
                                     'q_old': q_param_vec[-1], 'upper_bound_q': upper_bound_q, 'FLAG_lower_limits': False}

            HAL_FI_param.update(FLAG_constraints=HAL_FI_param.FLAG_constraints, query_constraints=query_constraints)

        # Time-Profiling
        if FLAG_time_profile is True:
            _profile_time0 = time.time()

        # Compute Optimal Query Distribution
        q_param = HAL_FI_param.optimal_query_distribution(A_cr, qs_model, p_ref=p_U)

        # Act like you're sampling from the ActionSpace and in turn update the n_shots left
        if env_qs is not None:
            X_q = A_cr.sample_action_space(env_qs, q_param, n_batches[k])

        if FLAG_time_profile is True:
            _profile_time1 = time.time()
            time_profile[k-1] = _profile_time1 - _profile_time0

        # Computer Fisher-Information Matrices
        FI_param_temp = HAL_FI_param.fisher_information_query(time_stamps_nd, xi_t, qs_model, q_param)
        FI_unif = HAL_FI_param.fisher_information_query(time_stamps_nd, xi_t, qs_model, p_U)

        # Update running Fisher-Information Matrices
        if HAL_FI_param.FLAG_constraints:
            if FLAG_lower_limits:
                FI_param_running = (N_p + n_batches[k]) * FI_param_temp
            else:
                FI_param_running = FI_param_running + n_batches[k] * FI_param_temp
        else:
            FI_param_running = FI_param_running + n_batches[k] * FI_param_temp

        # Update arrays of RMSE
        RMSE_param_array[k] = np.sqrt(np.trace(np.linalg.inv(FI_param_running)))
        TE_param_array[k] = np.trace(np.linalg.inv(FI_param_running) @ FI_unif)

        # Update other arrays
        N_p += n_batches[k]
        N_p_vec.append(N_p)
        q_param_vec.append(q_param)

        if FLAG_verbose:
            if k % int(max_k/5) == 0:
                print('k=%d' % (k))

    if FLAG_time_profile is True:
        results = {'rmse_param': RMSE_param_array, 'testing_error': TE_param_array, 'N_p': N_p_vec,
                   'q_J': q_J_vec, 'q_param': q_param_vec, 'time_profile': time_profile}
    else:
        results = {'rmse_param': RMSE_param_array, 'testing_error': TE_param_array, 'N_p': N_p_vec,
                   'q_J': q_J_vec, 'q_param': q_param_vec, 'A_cr': A_cr}

    return results


def policy_scalings(i_run, qs_model, moset, prepset, time_stamps, xi_t, n_batches, max_k, env_qs=None,
                    policy_type='rabi', FLAG_normalization=True, FLAG_noise=True,
                    FLAG_constraints = False, query_constraints_ref=None, FLAG_lower_limits=False,
                    max_iter_growth = 10, q_offline=None,
                    freq_convert=None, growth_time_stamps=None, growth_factor=2, type_spacing='linear',
                    FLAG_time_profile=False, solver_option='default', type_param='param'):
    """
    Function to calculate the optimal scalings given the parameters param_nd

    Setup SystemModel in the asymptotic regime
    Compute the optimal query distribution

    policy_type='rabi' or 'max_entropy'
    """
    print('Running with lower_limits on query constraints: %s' % FLAG_lower_limits)
    np.random.seed(10 * (i_run + 2))

    # Create some data structures
    time_stamps_nd = time_stamps / xi_t

    RMSE_param_array = np.zeros(max_k)

    if FLAG_time_profile is True:
        time_profile = np.zeros(max_k-1)

    # Running Fisher Information matrix
    FI_param_running = np.zeros(shape=(6, 6))

    # Initialize action space
    #A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps, xi_t, xi_param)
    A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps, xi_t,
                                        xi_J=qs_model.xi_J, xi_param=qs_model.xi_param, freq_convert=freq_convert)

    # Variable definitions
    N_config = A_cr.N_actions  # Number of different configurations/actions in the pool
    N_p_vec = []
    q_J_vec = []
    q_param_vec = []

    # Uniform probability distribution over the pool
    p_U = (1 / N_config) * np.ones(N_config)

    # Setup Active Learner
    HAL_FI_param = ActiveLearner(policy_type=policy_type,
                                 FLAG_normalization=FLAG_normalization,
                                 FLAG_noise=FLAG_noise,
                                 FLAG_constraints=FLAG_constraints,
                                 query_constraints=query_constraints_ref,
                                 type_param=type_param)

    # Act like you're sampling from the action space
    if env_qs is not None:
        X_p = A_cr.sample_action_space(env_qs, p_U, n_batches[0])

    # Computer Fisher-Information Matrices
    FI_param_temp = HAL_FI_param.fisher_information_query(time_stamps_nd, xi_t, qs_model, p_U)

    # Update running Fisher-Information Matrices
    FI_param_running = FI_param_running + n_batches[0] * FI_param_temp

    # Update arrays of RMSE
    RMSE_param_array[0] = np.trace(np.linalg.inv(FI_param_running))

    # Update other arrays
    N_p = n_batches[0]
    N_p_vec.append(N_p)
    q_param_vec.append(p_U)

    # For adaptive query space
    DT = np.abs(time_stamps[-1] - time_stamps[0])
    dt = np.abs(time_stamps[1] - time_stamps[0])

    n_iter_growth = 0
    k_growth = 1

    print('Using %s growing query space with %s spaced time stamps' % (growth_time_stamps, type_spacing))

    # ActiveLearning Run
    for k in range(1, max_k):
        # Update counter to check if it's time to grow the query space or not
        n_iter_growth += 1

        if growth_time_stamps is not None and n_iter_growth > max_iter_growth:
            k_growth += 1
            # Update the time_stamps
            print('Updating query space!')
            print(k_growth)

            if growth_time_stamps == 'linear':
                if type_spacing is 'linear':
                    t0_mod = time_stamps[-1] + dt
                    t1_mod = time_stamps[0] + k_growth * growth_factor * DT
                    tmid_mod = 0.5*(t0_mod + t1_mod)
                    time_stamps_mod = np.arange(tmid_mod, t1_mod, dt)
                else:
                    time_stamps_mod = np.arange(time_stamps[-1] + dt, time_stamps[0] + k_growth*growth_factor * DT, dt)

            elif growth_time_stamps == 'exponential':
                t0_mod = time_stamps[-1] + dt
                t1_mod = time_stamps[0] + (growth_factor ** (k_growth - 1)) * DT
                if type_spacing is 'linear':
                    tmid_mod = 0.5 * (t0_mod + t1_mod)
                    time_stamps_mod = np.arange(tmid_mod, t1_mod, dt)
                elif type_spacing is 'exponential':
                    tmid_mod = t1_mod - DT
                    time_stamps_mod = np.arange(tmid_mod, t1_mod, dt)
                else:
                    time_stamps_mod = np.arange(t0_mod, t1_mod, dt)
            else:
                raise RuntimeError('Passed invalid growth argument error')

            time_stamps = np.concatenate((time_stamps, time_stamps_mod), axis=0)
            time_stamps_nd = time_stamps / xi_t

            # Update the action space
            A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps_nd, xi_t,
                                                xi_J=qs_model.xi_J, xi_param=qs_model.xi_param,
                                                freq_convert=freq_convert)

            N_config = A_cr.N_actions

            # Update uniform query distribution as support has now changed
            p_U = (1 / N_config) * np.ones(N_config)

            # Reset the counter for growth
            n_iter_growth = 0

        # Update the ActiveLearner -- Query Constraints and (soon: adaptively growing query space)
        if FLAG_constraints:
            N_tot = N_p + n_batches[k]
            if FLAG_lower_limits:
                query_constraints = {'N_tot_old': N_p, 'N_tot': N_tot,
                                     'q_old': q_param_vec[-1], 'N_shots': query_constraints_ref['N_shots'], 'FLAG_lower_limits': True}
            else:
                upper_bound_q = [np.amin([A_cr.action_n_shots[ind]/A_cr.max_n_shots,1]) for ind in range(A_cr.N_actions)]
                query_constraints = {'N_tot_old': N_p, 'N_tot': N_tot,
                                     'q_old': q_param_vec[-1], 'upper_bound_q': upper_bound_q, 'FLAG_lower_limits': False}

            HAL_FI_param.update(FLAG_constraints=FLAG_constraints, query_constraints=query_constraints)

        # Time-Profiling
        if FLAG_time_profile is True:
            _profile_time0 = time.time()

        # Get the maximum entropy distribution
        if q_offline is None:
            if policy_type is 'max_entropy':
                q_param = qs_model.filter_actions_entropy(A_cr.action_space, time_stamps, xi_t=xi_t, FLAG_noise=FLAG_noise)
                q_param = q_param / np.sum(q_param)
            else:
                print('Using RB Distrn')
                q_param = qs_model.filter_actions_rb_crossings(A_cr.action_space, time_stamps, xi_t=xi_t, FLAG_noise=FLAG_noise)
                q_param = q_param/np.sum(q_param)
        else:
            q_param = np.copy(q_offline)

        # Act like you're sampling from the ActionSpace and in turn update the n_shots left
        if env_qs is not None:
            X_q = A_cr.sample_action_space(env_qs, q_param, n_batches[k])

        if FLAG_time_profile is True:
            _profile_time1 = time.time()
            time_profile[k-1] = _profile_time1 - _profile_time0

        # Computer Fisher-Information Matrices
        FI_param_temp = HAL_FI_param.fisher_information_query(time_stamps_nd, xi_t, qs_model, q_param)

        # Update running Fisher-Information Matrices
        if FLAG_constraints:
            if FLAG_lower_limits:
                FI_param_running = (N_p + n_batches[k]) * FI_param_temp
            else:
                FI_param_running = FI_param_running + n_batches[k] * FI_param_temp
        else:
            FI_param_running = FI_param_running + n_batches[k] * FI_param_temp

        # Update arrays of RMSE
        RMSE_param_array[k] = np.trace(np.linalg.inv(FI_param_running))

        # Update other arrays
        N_p += n_batches[k]
        N_p_vec.append(N_p)
        q_param_vec.append(q_param)

        if k % 5 == 0:
            print('k=%d' % (k))

    if FLAG_time_profile is True:
        results = {'rmse_param': RMSE_param_array, 'N_p': N_p_vec,
                   'q_J': q_J_vec, 'q_param': q_param_vec, 'time_profile': time_profile}
    else:
        results = {'rmse_param': RMSE_param_array, 'N_p': N_p_vec,
                   'q_J': q_J_vec, 'q_param': q_param_vec, 'A_cr': A_cr}

    return results


def plot_single_trend_exp_t_init_end(crb_U, nqueries_U,
                                     filename_savefig='scalings.eps', fig_label='Uniform'):
    rmse_U = np.sqrt(crb_U)

    ## Plotting

    # Uniform - Data
    plt.figure(figsize=(9, 9))
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.plot(nqueries_U, rmse_U, 'ro', label=fig_label)
    # plt.plot(nqueries_U, rmse_U, 'r:')

    # Uniform - Fit
    poly_U_init = np.polyfit(np.log(nqueries_U[0:6]), np.log(rmse_U[0:6]), 1)
    label_fit_U_init = "Slope = %0.2f" % (poly_U_init[0])
    plt.loglog(nqueries_U[0:6], np.exp(np.polyval(poly_U_init, np.log(nqueries_U[0:6]))), 'k--', label=label_fit_U_init)

    poly_U_end = np.polyfit(np.log(nqueries_U[-8:-1]), np.log(rmse_U[-8:-1]), 1)
    label_fit_U_end = "Slope = %0.2f" % (poly_U_end[0])
    plt.loglog(nqueries_U[-8:-1], np.exp(np.polyval(poly_U_end, np.log(nqueries_U[-8:-1]))), 'r-',
               label=label_fit_U_end)

    # plt.grid(True)
    plt.xlabel("Number of queries")
    plt.ylabel("RMSE")
    #plt.yticks([1e-2, 2e-2, 5e-2, 1e-1])
    #plt.gca().set_yticklabels([r'$10^{-2}$', r'$2 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$'])
    plt.legend(loc='lower left')
    plt.savefig(filename_savefig, bbox_inches='tight', dpi=300)


def plot_trends_single_learner(crb_U, crb_AL_FI, crb_AL_FIR,
                            nqueries_U, nqueries_AL_FI, nqueries_AL_FIR,
                            save_filename='comparison_trends_noise.png'):
    # Plot the trends
    color_lines = ['r-', 'r--', 'b-', 'b--', 'g-', 'g--']
    markers_lines = ['ro', 'rs', 'bo', 'bs', 'go', 'gs']
    labels_lines = ['Noiseless', 'Readout Noise', 'Readout + Control Noise']

    rmse_U = np.sqrt(crb_U)
    rmse_AL_FI = np.sqrt(crb_AL_FI)
    rmse_AL_FIR = np.sqrt(crb_AL_FIR)

    ## Plotting
    plt.figure(figsize=(9, 9))
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    # Uniform - Fits
    poly_U_init = np.polyfit(np.log(nqueries_U[0:6]), np.log(rmse_U[0:6]), 1)
    label_fit_U_init = "Slope = %0.2f" % (poly_U_init[0])
    plt.loglog(nqueries_U[0:6], np.exp(np.polyval(poly_U_init, np.log(nqueries_U[0:6]))), 'k--',
               label=label_fit_U_init)

    poly_U_end = np.polyfit(np.log(nqueries_U[-8:-1]), np.log(rmse_U[-8:-1]), 1)
    label_fit_U_end = "Slope = %0.2f" % (poly_U_end[0])
    plt.loglog(nqueries_U[-8:-1], np.exp(np.polyval(poly_U_end, np.log(nqueries_U[-8:-1]))), 'r-',
               label=label_fit_U_end)

    # Uniform - Data
    plt.plot(nqueries_U, rmse_U, markers_lines[0], label=labels_lines[0])

    # poly_U = np.polyfit(np.log(nqueries_U), np.log(rmse_U), 1)
    # label_fit_U = "Slope = %0.2f" % (poly_U[0])
    # plt.loglog(nqueries_U, np.exp(np.polyval(poly_U, np.log(nqueries_U))), 'k--', label=label_fit_U)

    # FIR - Data
    plt.plot(nqueries_AL_FIR, rmse_AL_FIR, markers_lines[2], label=labels_lines[1])
    # plt.plot(nqueries_AL_FIR, rmse_AL_FIR, 'b:')

    # FIR - Fits
    poly_AL_FIR_init = np.polyfit(np.log(nqueries_AL_FIR[0:6]), np.log(rmse_AL_FIR[0:6]), 1)
    label_fit_AL_FIR_init = "Slope = %0.2f" % (poly_AL_FIR_init[0])
    plt.loglog(nqueries_AL_FIR[0:6], np.exp(np.polyval(poly_AL_FIR_init, np.log(nqueries_AL_FIR[0:6]))), 'k:',
               label=label_fit_AL_FIR_init)

    poly_AL_FIR_end = np.polyfit(np.log(nqueries_AL_FIR[-8:-1]), np.log(rmse_AL_FIR[-8:-1]), 1)
    label_fit_AL_FIR_end = "Slope = %0.2f" % (poly_AL_FIR_end[0])
    plt.loglog(nqueries_AL_FIR[-8:-1], np.exp(np.polyval(poly_AL_FIR_end, np.log(nqueries_AL_FIR[-8:-1]))), 'b-',
               label=label_fit_AL_FIR_end)

    # FI - Data
    plt.plot(nqueries_AL_FI, rmse_AL_FI, markers_lines[4], label=labels_lines[2])
    # plt.plot(nqueries_AL_FI, rmse_AL_FI, 'g:')

    # FI - Fits
    poly_AL_FI_init = np.polyfit(np.log(nqueries_AL_FI[0:6]), np.log(rmse_AL_FI[0:6]), 1)
    label_fit_AL_FI_init = "Slope = %0.2f" % (poly_AL_FI_init[0])
    plt.loglog(nqueries_AL_FI[0:6], np.exp(np.polyval(poly_AL_FI_init, np.log(nqueries_AL_FI[0:6]))), 'k-',
               label=label_fit_AL_FI_init)

    poly_AL_FI_end = np.polyfit(np.log(nqueries_AL_FI[-8:-1]), np.log(rmse_AL_FI[-8:-1]), 1)
    label_fit_AL_FI_end = "Slope = %0.2f" % (poly_AL_FI_end[0])
    plt.loglog(nqueries_AL_FI[-8:-1], np.exp(np.polyval(poly_AL_FI_end, np.log(nqueries_AL_FI[-8:-1]))), 'g-',
               label=label_fit_AL_FI_end)

    plt.xlabel("Number of queries")
    plt.ylabel("RMSE")
    #plt.yticks([1e-2, 2e-2, 5e-2, 1e-1])
    #plt.gca().set_yticklabels([r'$10^{-2}$', r'$2 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$'])
    # plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.show()


def plot_trends_comparison_learners(rmse_learners, nqueries_learners,
                                    save_filename='comparison_trends_learners.png',
                                    labels_learners=['Uniform', r'HAL-FIR', r'HAL-FI'], figsize_plot=(12,12),
                                    FLAG_legend=True, FLAG_long_time_range=False, n_iters_end=5):
    # Plot the trends
    color_lines = ['r', 'b', 'g', 'c', 'm', 'y']
    marker_style = ['r+', 'bs', 'gd', 'co', 'mo', 'yo']
    line_styles = ['-', '--', ':', '-.']

    ## Plotting
    # To force xlabels and ylabels to use the right defined Font
    # Ref: https://stackoverflow.com/questions/11611374/sans-serif-font-for-axes-tick-labels-with-latex
    from matplotlib import rc
    rc('text.latex', preamble=r'\usepackage{sfmath}')

    plt.figure(figsize=figsize_plot)
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['text.usetex'] = True

    # Uniform - Data
    for ind in range(len(rmse_learners)):
        rmse_ind = rmse_learners[ind]
        nqueries_ind = nqueries_learners[ind]

        plt.plot(nqueries_ind, rmse_ind, marker_style[ind], fillstyle='none', label=labels_learners[ind])

        # RMSE of Learners (Fits) -- beginning and ending of trends
        poly_init = np.polyfit(np.log(nqueries_ind[0:6]), np.log(rmse_ind[0:6]), 1)
        label_fit_init = "Slope = %0.2f" % (poly_init[0])
        plt.plot(nqueries_ind[0:6], np.exp(np.polyval(poly_init, np.log(nqueries_ind[0:6]))),
                 color=color_lines[ind], linestyle='--', label=label_fit_init)

        poly_end = np.polyfit(np.log(nqueries_ind[-n_iters_end:]), np.log(rmse_ind[-n_iters_end:]), 1)
        label_fit_end = "Slope = %0.2f" % (poly_end[0])
        plt.plot(nqueries_ind[-n_iters_end:], np.exp(np.polyval(poly_end, np.log(nqueries_ind[-n_iters_end:]))),
                 color=color_lines[ind], linestyle='-', label=label_fit_end)

    # poly_U = np.polyfit(np.log(nqueries_U), np.log(rmse_U), 1)
    # label_fit_U = "Slope = %0.2f" % (poly_U[0])
    # plt.loglog(nqueries_U, np.exp(np.polyval(poly_U, np.log(nqueries_U))), 'k--', label=label_fit_U)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Number of queries")
    plt.ylabel("RMSE")
    plt.ylim([1.0e-4, 1.0e-1])

    if FLAG_long_time_range:
        plt.xticks([2e3, 5e3, 1e4, 2e4, 5e4])
        plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$',
                                   r'$5 \times 10^{4}$'])
    else:
        plt.xticks([2e3, 5e3, 1e4, 2e4])
        plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$'])

    #plt.yticks([1e-2, 2e-2, 5e-2, 1e-1])
    #plt.gca().set_yticklabels([r'$10^{-2}$', r'$2 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$'])
    # plt.grid(True)
    if FLAG_legend:
        if len(rmse_learners) < 4:
            plt.legend(loc='lower left')
        else:
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.savefig(save_filename, bbox_inches='tight', dpi=600)
    plt.show()


def plot_tvec_comparison_learners(tvec_learners, nqueries_learners, qs_T1, qs_T2, N_p_min, N_p_max,
                                  save_filename='comparison_tvec_trends_learners.png',
                                  labels_learners=['Uniform', r'HAL-FIR', r'HAL-FI'],
                                  figsize_plot=(20,7), FLAG_legend=True, FLAG_legend_outside=True):
    # Plot the trends
    color_lines = ['r-', 'g-', 'm-', 'r--', 'b--', 'g--', 'm--']
    marker_style = ['r+', 'gs', 'md', 'co', 'y^']
    line_styles = ['-', '--', ':', '-.']

    ## Plotting
    # To force xlabels and ylabels to use the right defined Font
    # Ref: https://stackoverflow.com/questions/11611374/sans-serif-font-for-axes-tick-labels-with-latex
    from matplotlib import rc
    rc('text.latex', preamble=r'\usepackage{sfmath}')

    fig = plt.figure(figsize=figsize_plot)
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['text.usetex'] = True

    # Uniform - Data
    for ind in range(len(tvec_learners)):
        rmse_ind = tvec_learners[ind]
        nqueries_ind = nqueries_learners[ind]

        plt.plot(nqueries_ind, rmse_ind, marker_style[ind], mew=1.5, fillstyle='none', label=labels_learners[ind])

    # T1 and T2 lines
    plt.hlines(y=np.amax(qs_T1), xmin=N_p_min, xmax=N_p_max, linewidth=1.5, linestyles='--', color='k')
    plt.hlines(y=np.amax(qs_T2), xmin=N_p_min, xmax=N_p_max, linewidth=1.5, linestyles=':', color='k')

    plt.text(1.8e4, 0.6*np.amax(qs_T1), r'$T_1$', fontsize=18)
    plt.text(1.8e4, 1.2*np.amax(qs_T2), r'$T_2$', fontsize=18)

    plt.xlabel("Number of queries")
    plt.ylabel("Max. Evolution Time t in QS [s]")
    plt.yscale('log')
    #plt.xscale('log')
    #plt.yticks([1e-2, 2e-2, 5e-2, 1e-1])
    #plt.gca().set_yticklabels([r'$10^{-2}$', r'$2 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$'])
    # plt.grid(True)

    if FLAG_legend:
        if FLAG_legend_outside:
            fig.legend(bbox_to_anchor=(0., -0.1, 1., 0.02), loc='lower center', ncol=2, labelspacing=0.)
        else:
            plt.legend(loc='best')

    plt.savefig(save_filename, bbox_inches='tight', dpi=600)
    plt.show()


def plot_learning_advantage(crb_U, crb_AL_FI,
                            nqueries_U, nqueries_AL_FI,
                            save_filename='hamiltonian_learning_advantage.png'):
    # Plot the trends
    color_lines = ['r-', 'r--', 'b-', 'b--', 'g-', 'g--']
    markers_lines = ['ro', 'rs', 'bo', 'bs', 'go', 'gs']
    labels_lines = ['Uniform (Random)', 'AL FIR (Test: p_u)', 'AL FI (Test: Unknown)']

    rmse_U = np.sqrt(crb_U)
    rmse_AL_FI = np.sqrt(crb_AL_FI)
    #rmse_AL_FIR = np.sqrt(crb_AL_FIR)

    ## Plotting

    # Uniform - Data
    plt.figure(figsize=(16, 10))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    plt.loglog(np.flip(rmse_U), np.flip(nqueries_U), '-ro', label='Passive')
    plt.loglog(np.flip(rmse_AL_FI), np.flip(nqueries_AL_FI), '-bo', label='HAL-FI')
    # plt.plot(nqueries_U, rmse_U, 'r:')

    # grab a reference to the current axes
    ax = plt.gca()
    # set the xlimits to be the reverse of the current xlimits
    ax.set_xlim(ax.get_xlim()[::-1])
    # call `draw` to re-render the graph
    plt.draw()

    plt.ylabel("Number of queries")
    plt.xlabel("RMSE")
    plt.legend(loc='lower right')
    plt.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.show()
