"""
Defines the oracles based on simulator or experimental data set
"""
import numpy as np
import sympy
import scipy.linalg
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
import collections
import random

from . import process_data
from ..quantum_system_models import quantum_device_models

# pauli matrices
si = np.array([ [1, 0], [0, 1] ])
sx = np.array([ [0, 1], [1, 0] ])
sy = np.array([ [0, -1j], [1j, 0] ])
sz = np.array([ [1, 0], [0, -1] ])

# hadamard
hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)


class Nature(object):
    """
    Represent Nature's knowledge of the (hidden) model Hamiltonian, providing
    a method to obtain sample measurement observable results.

    For the two cross-resonance coupled qubits.
    """

    def __init__(self, J, noise=None, expt_data=None, quantum_device='ibmq_boeblingen',
                 FLAG_simulator=True, FLAG_queryable_expt_data=False,
                 FLAG_readout_noise=False, FLAG_control_noise=False, FLAG_decoherence=False):
        """
        Inputs:
            Jix, Jiy, Jiz, Jzx, Jzy, Jzz = qubit coupling strengths (known to Nature, not to Physicist)

            noise is a dictionary with the following keys:
            readout: (r0,r1)
            imperfect_pulse_shaping: (teff0, teff1)
            decoherence: two-qubit decoherence model (assumed always!)

            Allowing nature to be a dataset itself
            FLAG_simulator = True for simulator and False for experimental dataset
            FLAG_queryable_data = True if expt data already in desired format for it to be queryable

            If False, exp_data better not be empty and must be in the ibm_data format

        TODO:
        1. Generalize depolarization models allowed
        2. Generalize handling of datasets
        """
        # Define Hamiltonian of quantum device oracle
        if len(J) != 6:
            raise ValueError("Expected 6 CR Hamiltonian parameters, got %d" % len(J))

        Jix, Jiy, Jiz, Jzx, Jzy, Jzz = J

        self.J = J

        self.Jix = Jix
        self.Jiy = Jiy
        self.Jiz = Jiz
        self.Jzx = Jzx
        self.Jzy = Jzy
        self.Jzz = Jzz

        self.IX = self.kron(si, sx)
        self.IY = self.kron(si, sy)
        self.IZ = self.kron(si, sz)
        self.ZX = self.kron(sz, sx)
        self.ZY = self.kron(sz, sy)
        self.ZZ = self.kron(sz, sz)

        self.hmat = (Jix * self.IX + Jiy * self.IY + Jiz * self.IZ +
                     Jzx * self.ZX + Jzy * self.ZY + Jzz * self.ZZ)

        self.basis = np.eye(4)  # basis vectors |0>, |1>, |2>, |3>
        self.psi0 = self.basis[:, 0]  # |0>

        # Noise is a part of the environment description too ;)

        # AND Giving the choice to not consider noise during estimation even if present
        self.FLAG_readout_noise = FLAG_readout_noise
        self.FLAG_control_noise = FLAG_control_noise
        self.FLAG_decoherence = FLAG_decoherence

        if noise is not None:
            if self.FLAG_readout_noise is True:
                self.readout_noise = noise['readout']
            else:
                self.readout_noise = np.array([0.0, 0.0])

            if self.FLAG_control_noise is True:
                self.imperfect_pulse_shaping = noise['imperfect_pulse_shaping']
            else:
                self.imperfect_pulse_shaping = None

            if self.FLAG_decoherence is True:
                self.decoherence_model = noise['decoherence_model']
            else:
                self.decoherence_model = None

        self.FLAG_simulator = FLAG_simulator
        if FLAG_simulator is False:
            print('Expt data simulator setup')
            if expt_data is None:
                raise ValueError('You requested Nature to be a dataset but you didnt give anything to work with!')
            else:
                self.FLAG_classification = expt_data['FLAG_classification']
                if FLAG_queryable_expt_data:
                    self.expt_data = expt_data
                    self.device = expt_data['device']
                else:
                    if 'device' in expt_data.keys():
                        self.device = expt_data['device']
                        if expt_data['device'] == 'ibmq_boeblingen':
                            self.expt_data = process_data.create_queryable_dataset_ibmq(expt_data,
                                                                                        FLAG_classification=self.FLAG_classification)
                        else:
                            print('Warning: unknown device key. Using default way of creating oracle from dataset')
                            self.expt_data = process_data.create_queryable_dataset(expt_data)
                    else:
                        # Do the default
                        self.device = quantum_device
                        self.expt_data = process_data.create_queryable_dataset(expt_data)

        elif FLAG_simulator is True:
            self.FLAG_classification = True
            self.device = quantum_device
            print('Simulator oracle setup')
        else:
            print('Improper FLAG_simulator. Try again')

    def print_info(self):
        if self.FLAG_simulator:
            print('Oracle: Simulator')
        else:
            print('Oracle: Experimental dataset')

        print('Noise Sources:')
        print('Readout Noise: FLAG=%r, Value=%s' %(self.FLAG_readout_noise, self.readout_noise))
        print('Control Noise: FLAG=%r' % self.FLAG_control_noise)
        print('Decoherence: FLAG=%r' % self.FLAG_decoherence)

    @staticmethod
    def kron(a, b):
        return np.array(scipy.linalg.kron(a, b))

    def sample_measurement_observable(self, observables, preparation, control, duration_pulse=0.0,
                                      nsamples=10, return_probabilities=False,
                                      return_state=False, return_symbolic=False):
        """
        observables = [M1, M0]
        preparation = [U1, U0]
        control = array of times
        """
        u10 = self.kron(*preparation)
        m10 = self.kron(*observables)

        samples = []
        kvals = range(4)
        #psi_in = u10 * self.psi0
        psi_in = u10 @ self.psi0

        if return_symbolic:
            Jix, Jiy, Jiz, Jzx, Jzy, Jzz, t = sympy.symbols("J_ix, J_iy, J_iz, J_zx, J_zy, J_zz, t", real=True)
            ham = (sympy.Matrix(self.IX) * Jix + sympy.Matrix(self.IY) * Jiy + sympy.Matrix(self.IZ) * Jiz
                   + sympy.Matrix(self.ZX) * Jzx + sympy.Matrix(self.ZY) * Jzy + sympy.Matrix(self.ZZ) * Jzz)
            psi = (sympy.Matrix(m10)
                   * sympy.exp(-1j * ham * t)
                   * sympy.Matrix(psi_in))
            pvec = [0] * 4
            for k in range(4):
                c = psi[k, 0]
                pvec[k] = (sympy.conjugate(c) * c).expand().simplify()
            return {'psi': psi, 'h': ham, 'probabilities': pvec}

        for t in control:
            uevol = scipy.linalg.expm(-1j * self.hmat * t)
            #psi = m10 * uevol * psi_in
            psi = m10 @ uevol @ psi_in

            #pvec = abs(np.array(psi.T)[0, :]) ** 2
            pvec = abs(psi) ** 2

            if self.FLAG_decoherence:
                # Modify the probabilities
                damping_array = self.decoherence_model(duration_pulse)
                pvec = damping_array[0] * pvec + damping_array[1]

            if return_state:  # immediately return (for first control value)
                return psi
            if return_probabilities:
                samples.append(pvec)
            else:
                pdf = scipy.stats.rv_discrete(name="psi", values=(kvals, pvec))
                samples.append(pdf.rvs(size=nsamples))

        return np.array(samples)

    def sample_meas_obs_noise(self, observables, preparation, control, duration_pulse=0.0, nsamples=10,
                              return_probabilities=False, return_state=False):
        """
        Gives out samples and measurements with readout noise
        Imperfect pulse shaping can be cheated by giving shifted time stamps into this

        Should be merged with above function later
        observables = [M1, M0]
        preparation = [U1, U0]
        control = array of times
        """
        u10 = self.kron(*preparation)
        m10 = self.kron(*observables)

        samples = []
        kvals = range(4)
        #psi_in = u10 * self.psi0
        psi_in = u10 @ self.psi0

        for t in control:
            uevol = scipy.linalg.expm(-1j * self.hmat * t)
            #psi = m10 * uevol * psi_in
            psi = m10 @ uevol @ psi_in

            #pvec = abs(np.array(psi.T)[0, :]) ** 2
            pvec = abs(psi) ** 2

            if self.FLAG_decoherence:
                # Modify the probabilities
                damping_array = self.decoherence_model(duration_pulse)
                pvec = damping_array[0] * pvec + damping_array[1]

            if return_state:  # immediately return (for first control value)
                return psi
            if return_probabilities:
                samples.append(pvec)
            else:
                pdf = scipy.stats.rv_discrete(name="psi", values=(kvals, pvec))

                # Samples before readout noise applied
                ideal_samples = pdf.rvs(size=nsamples)

                # Get number of 0s and 1s in the above samples
                n_zeros = np.sum(ideal_samples == 0)
                n_ones = np.sum(ideal_samples == 1)

                # Generate random numbers to decide flip and
                # then compare against p(flip|0) or p(flip|1) to get array of 0s (no flip) and 1s (flip)

                rnd_flip0 = np.random.rand(n_zeros)
                FLAG_flip0 = rnd_flip0 < self.readout_noise[0]
                FLAG_flip0 = FLAG_flip0.astype(int)

                rnd_flip1 = np.random.rand(n_ones)
                FLAG_flip1 = rnd_flip1 < self.readout_noise[1]
                FLAG_flip1 = FLAG_flip1.astype(int)

                # Now flip stuff using the flag
                noisy_samples = np.copy(ideal_samples)
                noisy_samples[ideal_samples == 0] = ideal_samples[ideal_samples == 0] + FLAG_flip0
                noisy_samples[ideal_samples == 1] = ideal_samples[ideal_samples == 1] + FLAG_flip1
                noisy_samples = np.mod(noisy_samples,2)

                samples.append(noisy_samples)

        return np.array(samples)

    def smcg_sample(self, control, **opts):
        """
        Return set of samples for time specified by control, for three measurement
        observables and two preparation operators, as specified by Sheldon,
        Magesan, Chow, and Gambetta.
        """

        zp = np.matrix([[1, 0], [0, 1j]])

        # Set of measurement operators
        moset = {0: si, 1: hadamard, 2: hadamard * zp}
        prepset = {0: [si, si], 1: [sx, si]}
        sdat = []
        for m in [1, 2, 0]:
            for u in [0, 1]:
                uop = prepset[u]
                mop = [uop[0], moset[m]]
                samples = self.sample_measurement_observable(mop, uop, control, **opts)
                sdat.append({'u': u, 'm': m, 'samples': samples})
        return sdat

    def sample_expt_data(self, ind_action, nsamples=1):
        """
        1. Pop random measurement outcome corresponding to ind_action
        2. Update the size of that array in the oracle

        :param ind_action:
        :param nsamples:
        :return:
        """
        # Get number of samples remaining for particular action
        nsamples_action_i = self.expt_data['n_samples_actions'][ind_action]

        samples = []

        if self.FLAG_classification is False:
            samples_p0 = []
            samples_p1 = []

        for ind_sample in range(nsamples):
            if nsamples_action_i > 0:
                # Get a random index to sample
                rand_sample = np.random.randint(0, high=nsamples_action_i)

                samples.append(self.expt_data[ind_action].pop(rand_sample))
                self.expt_data['n_samples_actions'][ind_action] -= 1

                if self.FLAG_classification is False:
                    samples_p0.append(self.expt_data['samples_p0'][ind_action].pop(rand_sample))
                    samples_p1.append(self.expt_data['samples_p1'][ind_action].pop(rand_sample))

            else:
                print('Have used up all the samples from this action!')

        if self.FLAG_classification is False:
            return samples, samples_p0, samples_p1
        else:
            return samples


class Action_Space(object):
    """
    Define the space of all possible actions/configurations (queries)

    For the two cross-resonance coupled qubits.
    """

    def __init__(self, moset, prepset, tset, xi_t, xi_J, n_shots=1e8, xi_param=None, freq_convert=None):
        """
        Function to initialize the pool (set of all possible actions/queries)

        moset = Set of measurement operators (dictionary)
        prepset = Set of preparation operators (dictionary)
        tset = Set of time-stamps that can be queried (non-dimensional)
        xi_t = scalings for time
        xi_param = scalings for parameter set (Should include xi_J later)

        TO DO: Handling both xi_param and xi_J

        action_space is array of [m,u,t] where
        m and u correspond to keys in moset and prepset
        t are the non-dimensional times from tset
        """
        self.moset = moset
        self.prepset = prepset
        self.tset = tset
        self.xi_t = xi_t
        self.xi_J = xi_J
        self.xi_param = xi_param

        # Time-stamps and frequency information of ActionSpace
        time_stamps = tset * xi_t
        _dt = (time_stamps[-1] - time_stamps[0]) / (len(time_stamps) - 1)
        _sample_freq = 1.0 / _dt

        if freq_convert is None:
            freq_convert = _sample_freq * 2.0 * np.pi / len(time_stamps)

        action_space = []
        action_n_shots = [] # Number of shots available for each action

        ## HACK! Replace below with pandas objects later
        ## HACK! Note the rounding required!!! hardcoded at the moment ... Lines 265 and 310,312
        dict_action_space = {}
        dict_time_to_index = {}
        dict_index_to_time = {}

        # Should probably vectorize this -- And replace the dict_action_space thingy

        ## TO DO: The mappings between indices and time-stamps only works for the fixed action query space
        #t_nd0 = tset[0]
        #dt_nd = np.abs(tset[1]-tset[0])
        # Note that dt in tset is not constant and hence ind_t = np.floor((t-t0)/dt).astype(int) wouldn't be reliable!

        count_time = 0
        for t_nd in tset:
            ind_t = count_time
            dict_index_to_time.update({ind_t: t_nd})

            # str is important. Number as it is was confusing line 315
            dict_time_to_index.update({str(np.round(t_nd,3)): ind_t})

            for m in moset.keys():
                for u in prepset.keys():
                    action_space.append([m,u,t_nd])
                    action_n_shots.append(n_shots)

                    # Dictionary: key is tuple of (m,u,t) and value is [nsamples, n0]
                    dict_action_space.update({(m, u, ind_t): [0, 0]})

            count_time += 1

        self.action_space = action_space
        self.dict_action_space = dict_action_space

        self.action_n_shots = np.asarray(action_n_shots, dtype=int)
        self.max_n_shots = n_shots

        self.dict_time_to_index = dict_time_to_index
        self.dict_index_to_time = dict_index_to_time

        # Number of actions
        self.N_actions_M = len(moset.keys())    # Just M
        self.N_actions_U = len(prepset.keys())  # Just U
        self.N_actions_t = len(tset)    # Just time
        self.N_actions = len(action_space)  # Total = _M*_U*_t

        # Property of sampling from action space - freq_convert (lines 553-556 of process_data.py)
        self.freq_convert = freq_convert
        self.sample_freq = _sample_freq

    def update_dict_action_space(self, X_p, n_shots_old=None, N_actions_old=None):
        """
        Function to deal with the case when actions are not as uniform and as nice collected as above

        Usage: Despite the notation X_p, this should actually be the set of new queries as dict_action_space already
        contains values from before

        :param A_cr:
        :param X_q:
        :param xi_t:
        :return:
        """
        t_nd = self.tset

        # Get list of actions in data -- Note that value for 'tvec' is not normalized but is for 'time'
        if 'time' in X_p.keys():
            time_samples = np.round(np.asarray(X_p['time'], dtype=float), 3)
        else:
            time_samples = np.round(np.asarray(X_p['tvec'], dtype=float) / self.xi_t, 3)

        # Should update the dictionary below to the action with the closest time
        list_actions = [((X_p['mvec'][ind], X_p['uvec'][ind], self.dict_time_to_index[str(time_samples[ind])]),
                         X_p['samples'][ind]) for ind in range(len(X_p['samples']))]

        # Update the A_cr dict
        #import pdb; pdb.set_trace()
        for x_i, y_i in list_actions:
            self.dict_action_space[x_i][0] += 1
            self.dict_action_space[x_i][1] += (1 - y_i)

        # Update the number of shots
        if n_shots_old is not None:
            self.action_n_shots[0:N_actions_old] = np.copy(n_shots_old)

    @staticmethod
    def query_to_config(m, u):
        return 2*m + u

    def rabi_data(self, env_quantum_sys=None):
        """
        Nature (environment) described by env_quantum_sys

        :param env_quantum_sys:
        :return:
        """

        ## Get rabi oscillations of the data
        # dicts in python are now ordered but shouldn't be relied on???
        # Ref: https://pymotw.com/3/collections/ordereddict.html

        p0 = np.zeros(len(self.dict_action_space.keys()))
        counter = 0

        for key_i in self.dict_action_space.keys():
            # _temp = [nsamples, n0]
            _temp = self.dict_action_space[key_i]

            if _temp[0] == 0:
                p0[counter] = np.nan
            else:
                p0[counter] = _temp[1] / (_temp[0] + 1e-16)

            counter += 1

        p0 = 2 * p0 - 1

        pvec = np.zeros((6, len(self.tset)))
        for i in range(6):
            pvec[i, :] = p0[i::6]

        # NOTE: We don't divide by (1-r0-r1) because the estimation procedure should account for this
        # if env_quantum_sys.FLAG_readout_noise:
        #     r0, r1 = env_quantum_sys.readout_noise
        #     pvec = pvec / (1.0 - r0 - r1)

        return pvec

    def nshots_rabi_data(self, env_quantum_sys):
        """
        Nature (environment) described by env_quantum_sys

        :param env_quantum_sys:
        :return:
        """

        ## Get rabi oscillations of the data
        # dicts in python are now ordered but shouldn't be relied on???
        # Ref: https://pymotw.com/3/collections/ordereddict.html

        n_shots = np.zeros(len(self.dict_action_space.keys()))
        n0_shots = np.zeros(len(self.dict_action_space.keys()))

        counter = 0
        for key_i in self.dict_action_space.keys():
            # _temp = [nsamples, n0]
            _temp = self.dict_action_space[key_i]
            n_shots[counter] = _temp[0]
            n0_shots[counter] = _temp[1]
            counter += 1

        nshots_vec = np.zeros((6, len(self.tset)))
        n0shots_vec = np.zeros((6, len(self.tset)))
        for i in range(6):
            nshots_vec[i, :] = n_shots[i::6]
            n0shots_vec[i, :] = n0_shots[i::6]

        return np.asarray(nshots_vec, dtype=int), np.asarray(n0shots_vec, dtype=int)

    def nonempty_actions(self, nshots_actions_temp=None, nshots_threshold=1):
        """
        Function that returns indices of queries/actions that are non-empty and/or (> nshots_threshold)
        and corresponding action-space
        """
        if nshots_actions_temp is None:
            nshots_actions_temp = self.action_n_shots

        valid_action_set = np.where(nshots_actions_temp > nshots_threshold)[0]

        return valid_action_set

    def filtered_actions(self, FLAG_uncertainty_filtering=False):
        """
        At the moment, filter actions to make sure constraints that are satisfied
        ahead of time
        """
        valid_action_set_temp = self.nonempty_actions()

        filtered_actions_list = []

        for ind in range(len(valid_action_set_temp)):
            filtered_actions_list.append(self.action_space[valid_action_set_temp[ind]])

        return filtered_actions_list

    def actions_pruning(self, samples_actions):
        """
        Prunes the action set that is to be queried and makes sure that can indeed satisfy query constraints
        (e.g.. nshots available)
        """
        # get collection (julia functionality) of samples_actions -- inefficient (may need to be replaced by pandas)
        # ref: https://stackoverflow.com/questions/23240969/python-count-repeated-elements-in-the-list/23240989
        dict_actions_q = dict(collections.Counter(samples_actions))

        # convert above dictionary to shots made for each query
        # ref: https://stackoverflow.com/questions/23668509/dictionary-keys-and-values-to-separate-numpy-arrays
        ind_actions_q = np.fromiter(dict_actions_q.keys(), dtype=int)
        nshots_actions_q = np.fromiter(dict_actions_q.values(), dtype=int)

        # get queries for which number of shots requested exceeded that allowed and set
        nshots_left_over = self.action_n_shots[ind_actions_q] - nshots_actions_q
        ind_shots_exceeded = np.where(nshots_left_over < 0)[0]

        # set actions_q according to what can be done
        nshots_actions_q[ind_shots_exceeded] += nshots_left_over[ind_shots_exceeded]

        # nshots left over that need to have associated actions
        nshots_not_set = -1*np.sum(nshots_left_over[ind_shots_exceeded])

        # update nshots left over now
        nshots_left_over = self.action_n_shots[ind_actions_q] - nshots_actions_q

        # start setting up an array containing the actions to be made
        samples_actions_pruned = []
        for ind in range(len(nshots_actions_q)):
            ind_query = ind_actions_q[ind]
            n_query_i = nshots_actions_q[ind]
            samples_actions_pruned.extend([ind_query] * n_query_i)

        # sample shots not set uniformly from left over sequentially
        nshots_actions_temp = np.copy(self.action_n_shots)
        nshots_actions_temp[ind_actions_q] = nshots_left_over

        for ind_shot in range(nshots_not_set):
            # get non-empty actions (indices of them)
            valid_action_set_temp = self.nonempty_actions(nshots_actions_temp=nshots_actions_temp)

            # choose one uniformly from above
            ind_query = random.choice(valid_action_set_temp)
            samples_actions_pruned.extend([ind_query])
            nshots_actions_temp[ind_query] -= 1

        return samples_actions_pruned

    def sample_action_space_quantum_dev_model(self, quantum_sys_model, p_query, N_batch,
                                              actions_list=None, FLAG_query=True):
        """
        Sampling from probability distributions prescribed by quantum_device_models submodule
        p_query = pdf distribution of the query
        N_batch = number of batch of queries to issue

        FLAG_query to indicate if probability distribution being used or number of samples.
        Var of interest here is p_query p_query

        Usage: FLAG_query is true and p_query[i] = 0.5 then ith query is sampled with prob 0.5
        FLAG_query is false and p_query[i] = 5 then ith query is sampled 5 times

        returns actions and corresponding samples
        """

        # Sample from the discrete space of actions
        if actions_list is None:
            actions_list = np.arange(self.N_actions)
            N_valid_actions = self.N_actions
        else:
            N_valid_actions = len(actions_list)

        if FLAG_query:
            p_actions = scipy.stats.rv_discrete(name="p_actions", values=(actions_list, p_query))
            samples_actions_query = p_actions.rvs(size=N_batch)
            samples_actions = self.actions_pruning(samples_actions_query)
        else:
            if N_batch % N_valid_actions != 0:
                print('Working in non-query mode. N_batch should be a multiple of N_actions')
                return None

            samples_actions = []
            p_query = np.asarray(p_query, dtype=int)
            for ind_query in range(N_valid_actions):
                # number of samples associated with query_i
                n_query_i = p_query[ind_query]

                # Ref: https://stackoverflow.com/questions/4654414/python-append-item-to-list-n-times
                samples_actions.extend([actions_list[ind_query]]*n_query_i)


        # Apply query corresponding to each action and sample (INEFFICIENT)
        # Can be sped up by noting that there are a few actions which are repeated over and over
        mvec_query = []
        uvec_query = []
        config_query = []
        tvec_query = []
        samples_query_tmp = np.zeros(N_batch).astype(np.int16)

        kvals = range(4)
        #import pdb; pdb.set_trace()

        for i in range(N_batch):
            ind_action_i = samples_actions[i]
            action_i = self.action_space[ind_action_i]
            m_i, u_i, t_i = action_i

            n_config_temp = 2*m_i + u_i
            pvec_temp = quantum_sys_model.model_probabilities(n_config_temp, t_i*self.xi_t, FLAG_noise=False)

            pdf_temp = scipy.stats.rv_discrete(name="psi", values=(kvals, pvec_temp))
            samples_query_tmp[i] = pdf_temp.rvs(size=1)

            mvec_query.append(m_i)
            uvec_query.append(u_i)
            config_query.append(2*m_i + u_i)
            tvec_query.append(t_i)

            # Update shots left for that query
            self.action_n_shots[ind_action_i] -= 1

        # Assuming we only observe the state of qubit 2
        samples_query = np.array([int(bin(s)[-1]) for s in samples_query_tmp])

        X_q = {'samples': samples_query, 'config': config_query, 'time': tvec_query,
               'tvec': np.array(tvec_query) * self.xi_t,
               'mvec': mvec_query, 'uvec': uvec_query, 'actions': samples_actions,
               'xi_t': self.xi_t, 'xi_J': self.xi_J, 'FLAG_classification': True,
               'misclassif_error': quantum_sys_model.readout_noise, 'device': 'ibmq_boeblingen',
               'time_stamps': self.tset, 'freq_convert': self.freq_convert}

        # sample_actions included in X_q for debugging purposes

        return X_q

    def sample_action_space(self, env_quantum_sys, p_query, N_batch, actions_list=None, FLAG_query=True):
        """
        Nature (environment) described by env_quantum_sys
        p_query = pdf distribution of the query
        N_batch = number of batch of queries to issue

        FLAG_query to indicate if probability distribution being used or number of samples.
        Var of interest here is p_query

        Usage: FLAG_query is true and p_query[i] = 0.5 then ith query is sampled with prob 0.5
        FLAG_query is false and p_query[i] = 5 then ith query is sampled 5 times

        returns actions and corresponding samples
        """
        # Sample from the discrete space of actions
        if actions_list is None:
            actions_list = np.arange(self.N_actions)
            N_valid_actions = self.N_actions
        else:
            N_valid_actions = len(actions_list)

        if FLAG_query:
            p_actions = scipy.stats.rv_discrete(name="p_actions", values=(actions_list, p_query))
            samples_actions_query = p_actions.rvs(size=N_batch)
            samples_actions = self.actions_pruning(samples_actions_query)
        else:
            if N_batch % N_valid_actions != 0:
                print('Working in non-query mode. N_batch should be a multiple of N_actions')
                return None

            samples_actions = []
            p_query = np.asarray(p_query, dtype=int)
            for ind_query in range(N_valid_actions):
                # number of samples associated with query_i
                n_query_i = p_query[ind_query]

                # Ref: https://stackoverflow.com/questions/4654414/python-append-item-to-list-n-times
                samples_actions.extend([actions_list[ind_query]]*n_query_i)

        # Apply query corresponding to each action and sample (INEFFICIENT)
        # Can be sped up by noting that there are a few actions which are repeated over and over
        mvec_query = []
        uvec_query = []
        config_query = []
        tvec_query = []
        samples_query_tmp = np.zeros(N_batch).astype(np.int16)

        if env_quantum_sys.FLAG_classification is False:
            samples_p0_query_tmp = np.zeros(N_batch).astype(np.int16)
            samples_p1_query_tmp = np.zeros(N_batch).astype(np.int16)

        if env_quantum_sys.FLAG_simulator is True:
            for i in range(N_batch):
                ind_action_i = samples_actions[i]
                action_i = self.action_space[ind_action_i]
                m_i, u_i, t_i = action_i

                # Even though below is an array of a single element,
                # sample_measurement_observable below requires control to be an array
                if env_quantum_sys.FLAG_control_noise:
                    control = [t_i*self.xi_t + env_quantum_sys.imperfect_pulse_shaping[u_i]] # Imitating pulse edge effects
                else:
                    control = [t_i*self.xi_t]

                # Update set of samples (results) of queries
                mop = self.moset[m_i]
                uop = self.prepset[u_i]

                if env_quantum_sys.FLAG_readout_noise:
                    samples_query_tmp[i] = env_quantum_sys.sample_meas_obs_noise(mop, uop, control,
                                                                                 duration_pulse=t_i*self.xi_t,
                                                                                 nsamples=1)
                else:
                    samples_query_tmp[i] = env_quantum_sys.sample_measurement_observable(mop, uop, control,
                                                                                         duration_pulse=t_i*self.xi_t,
                                                                                         nsamples=1)

                mvec_query.append(m_i)
                uvec_query.append(u_i)
                config_query.append(2*m_i + u_i)
                tvec_query.append(t_i)

                # Update shots left for that query
                self.action_n_shots[ind_action_i] -= 1

            # Assuming we only observe the state of qubit 2
            samples_query = np.array([int(bin(s)[-1]) for s in samples_query_tmp])

        elif env_quantum_sys.FLAG_simulator is False:
            for i in range(N_batch):
                ind_action_i = samples_actions[i]
                action_i = self.action_space[ind_action_i]
                m_i, u_i, t_i = action_i

                # Update set of samples (results) of queries
                if env_quantum_sys.FLAG_classification is False:
                    s_temp, p0_temp, p1_temp = env_quantum_sys.sample_expt_data(ind_action_i, nsamples=1)
                    samples_query_tmp[i] = s_temp[0]
                    samples_p0_query_tmp[i] = p0_temp[0]
                    samples_p1_query_tmp[i] = p1_temp[0]
                else:
                    samples_query_tmp[i] = env_quantum_sys.sample_expt_data(ind_action_i, nsamples=1)[0]

                # Update shots left for that query
                self.action_n_shots[ind_action_i] -= 1

                mvec_query.append(m_i)
                uvec_query.append(u_i)
                config_query.append(2*m_i + u_i)
                tvec_query.append(t_i)

            # Assuming we only observe the state of qubit 2
            samples_query = np.copy(samples_query_tmp)

            if env_quantum_sys.FLAG_classification is False:
                samples_p0_query = np.copy(samples_p0_query_tmp)
                samples_p1_query = np.copy(samples_p1_query_tmp)

        else:
            print('Type of system oracle unknown. Set FLAG_simulator')
            return None

        X_q = {'samples': samples_query, 'config': config_query, 'time': tvec_query,
               'tvec': np.array(tvec_query) * self.xi_t,
               'mvec': mvec_query, 'uvec': uvec_query, 'actions': samples_actions,
               'xi_t': self.xi_t, 'xi_J': self.xi_J, 'FLAG_classification': env_quantum_sys.FLAG_classification,
               'misclassif_error': env_quantum_sys.readout_noise, 'device': env_quantum_sys.device,
               'time_stamps': self.tset, 'freq_convert': self.freq_convert}

        if env_quantum_sys.FLAG_classification is False:
            X_q.update({'samples_p0': samples_p0_query, 'samples_p1': samples_p1_query})

        # Only if we are using the experimental data as oracle, consider setting FLAG for using decoherence later
        if not env_quantum_sys.FLAG_simulator:
            if 'FLAG_decoherence' in env_quantum_sys.expt_data.keys():
                X_q.update({'FLAG_decoherence': env_quantum_sys.expt_data['FLAG_decoherence']})

        # sample_actions included in X_q for debugging purposes

        return X_q

    def merge_datasets(self, X_p, X_q):
        """
        :param X_p: data obtained from the pool so far
        :param X_q: data most recently obtained through queries
        :return: merged datasets
        """
        # # Keys for which the terms should be same
        # keys_same = ['xi_t', 'xi_J']
        # keys_same = ['xi_t', 'xi_param']
        # for key_i in keys_same:
        #     if X_p[key_i].any() != X_q[key_i].any():
        #         print('Something is off with key %s', key_i)

        # Also add error if key not in one of the datasets
        X = {'xi_t': self.xi_t, 'time_stamps': self.tset, 'freq_convert': self.freq_convert,
             'FLAG_classification': X_p['FLAG_classification'], 'misclassif_error': X_p['misclassif_error'],
             'device': X_p['device']}

        if 'xi_param' in X_p.keys():
            X.update({'xi_param': X_p['xi_param']})
        else:
            X.update({'xi_J': X_p['xi_J']})

        if 'FLAG_decoherence' in X_p.keys():
            X_q.update({'FLAG_decoherence': X_p['FLAG_decoherence']})

        # Keys for which data should be merged
        if X_p['FLAG_classification'] is False:
            keys_data = ['samples', 'samples_p0', 'samples_p1', 'mvec', 'uvec', 'tvec', 'time', 'config', 'actions']
        else:
            keys_data = ['samples', 'mvec', 'uvec', 'tvec', 'time', 'config', 'actions']

        for key_i in keys_data:
            X.update({key_i: np.concatenate((X_p[key_i], X_q[key_i]))})

        return X

    def visualize_query_distrn(self, p_query, cmax=0.15, FLAG_indicate_query_space=True,
                               save_plot=False, title_plot=None):
        """
        :param p_query: query distribution
        :param save_plot:
        :param title_plot:
        FLAG_indicate_query_space: whether to indicate the usual query space used
        :return:
        """
        q_plot = p_query.reshape([self.N_actions_t, self.N_actions_M*self.N_actions_U]).T

        # To force xlabels and ylabels to use the right defined Font
        # Ref: https://stackoverflow.com/questions/11611374/sans-serif-font-for-axes-tick-labels-with-latex
        from matplotlib import rc
        rc('text.latex', preamble=r'\usepackage{sfmath}')

        plt.figure(2, figsize=(14, 6))
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams['font.sans-serif'] = "Helvetica"
        plt.rcParams['figure.titlesize'] = 24
        plt.rcParams['axes.labelsize'] = 24
        plt.rcParams['axes.titlesize'] = 24
        plt.rcParams['legend.fontsize'] = 18
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['text.usetex'] = True

        img = plt.imshow(q_plot, interpolation='nearest', aspect='auto')
        img.set_cmap('Blues')
        plt.colorbar()
        plt.clim(0, cmax)

        plt.xlabel(r"t ($\times 10^{-7}$s)")
        plt.ylabel(r"($M,U$)")

        if FLAG_indicate_query_space:
            ylabels_plot = [r'', r'($M_{\langle X \rangle}, U_0$)', r'($M_{\langle X \rangle}, U_1$)',
                            r'($M_{\langle Y \rangle}, U_0$)', r'($M_{\langle Y \rangle}, U_1$)',
                            r'($M_{\langle Z \rangle}, U_0$)', r'($M_{\langle Z \rangle}, U_1$)']

            fig_axis = plt.gca()
            fig_axis.set_yticklabels(ylabels_plot)

            xlocs, xlabels = plt.xticks()  # Get locations and labels

            # xlabels has empty strings in the very beginning and end
            # xlocs includes the leftmost and rightmost

            n_xlabels = 10
            xlocs = np.linspace(0, len(self.tset)-1, n_xlabels, dtype=int)
            time_info = self.tset[xlocs]
            #xlabels_plot = [r'']
            xlabels_plot = []
            for ind in range(len(time_info)):
                xlabels_plot.append(str(np.round(time_info[ind],1)))

            #xlabels_plot.append(r'')
            plt.xticks(xlocs, xlabels_plot)  # Set locations and labels

        if save_plot:
            if title_plot is None:
                print('Title not provided, saving as viz1.pdf')
                title_plot = 'viz1.pdf'

            plt.savefig(title_plot, bbox_inches='tight', dpi=600)

        plt.show()


# Quick helper function to get testing dataset
def get_testing_dataset(env_quantum_sys, A_cr_train):
    """
    Get the testing dataset after creation of training dataset

    This is not a method of ActionSpace to ensure that we don't accidentally change the object of interest

    Inputs:
        A_cr_train: ActionSpace associated with the training dataset
        env_quantum_sys: Oracle

    Returns:
        X_test: Testing dataset
        A_cr_test: ActionSpace associated with the testing dataset
    TODO: Needs to be generalized for adaptively growing time where experimental datasets are stitched over rounds
    """
    if env_quantum_sys.FLAG_simulator is not False:
        raise RuntimeError('Type of system oracle doesnt allow for this! Set FLAG_simulator!')

    # Go through list of remaining actions and create testing dataset
    mvec_test = []
    uvec_test = []
    config_test = []
    tvec_test = []

    N_remaining_actions = sum(env_quantum_sys.expt_data['n_samples_actions'])
    samples_test = np.zeros(N_remaining_actions).astype(np.int16)

    if env_quantum_sys.FLAG_classification is False:
        samples_p0_test = np.zeros(N_remaining_actions)
        samples_p1_test = np.zeros(N_remaining_actions)

    # Loop over all the actions
    counter = 0
    actions_n_shots_test = np.zeros(A_cr_train.N_actions) # Will hold the actions for A_cr_test
    for action_i in range(A_cr_train.N_actions):
        n_remaining_actions_i = env_quantum_sys.expt_data['n_samples_actions'][action_i]
        actions_n_shots_test[action_i] = A_cr_train.max_n_shots - n_remaining_actions_i
        if n_remaining_actions_i > 0:
            tuple_action_i = A_cr_train.action_space[action_i]
            m_i, u_i, t_i = tuple_action_i

            # sample the leftovers
            for ind_left in range(n_remaining_actions_i):
                # Update set of samples
                if env_quantum_sys.FLAG_classification is False:
                    s_temp, p0_temp, p1_temp = env_quantum_sys.sample_expt_data(action_i, nsamples=1)
                    samples_test[counter] = s_temp[0]
                    samples_p0_test[counter] = p0_temp[0]
                    samples_p1_test[counter] = p1_temp[0]
                else:
                    samples_test[counter] = env_quantum_sys.sample_expt_data(action_i, nsamples=1)[0]

                # Update samples in testing dataset
                mvec_test.append(m_i)
                uvec_test.append(u_i)
                config_test.append(2 * m_i + u_i)
                tvec_test.append(t_i)

                counter += 1

    X_test = {'samples': samples_test, 'config': config_test, 'time': tvec_test,
              'tvec': np.array(tvec_test) * A_cr_train.xi_t,
              'mvec': mvec_test, 'uvec': uvec_test, 'actions': samples_test,
              'xi_t': A_cr_train.xi_t, 'xi_J': A_cr_train.xi_J,
              'FLAG_classification': env_quantum_sys.FLAG_classification,
              'misclassif_error': env_quantum_sys.readout_noise, 'device': env_quantum_sys.device,
              'time_stamps': A_cr_train.tset, 'freq_convert': A_cr_train.freq_convert}

    if env_quantum_sys.FLAG_classification is False:
        X_test.update({'samples_p0': samples_p0_test, 'samples_p1': samples_p1_test})
        
    # Create ActionSpace associated with testing dataset -- copy some of the A_cr_train attributes
    A_cr_test = Action_Space(A_cr_train.moset, A_cr_train.prepset, A_cr_train.tset,
                             A_cr_train.xi_t, A_cr_train.xi_J, n_shots=A_cr_train.max_n_shots,
                             xi_param=A_cr_train.xi_param, freq_convert=A_cr_train.freq_convert)

    A_cr_test.update_dict_action_space(X_test, n_shots_old=actions_n_shots_test, N_actions_old=A_cr_train.N_actions)

    return X_test, A_cr_test


def model_probabilities_ham(env_qs, moset, prepset, J_num, uvec, mvec, tvec):
    """
    Return p_um(t) = smcg model probability of different outcomes |0>, |1>, |2>, |3> for
    preparation u and measurement m at time t. Use general expressions for hamiltonian
    for evaluation.
    u in [0,1]
    m in [0,1,2]
    """
    prob_samples = np.zeros((mvec.size, 4))

    for i in range(0, mvec.size):
        m = mvec[i]
        u = uvec[i]
        t = tvec[i]

        # Note the difference in how mop is defined compared to TwoQubitCoupling or CRModel cases
        uop = prepset[u]
        mop = moset[m]

        u10 = env_qs.kron(*uop)
        m10 = env_qs.kron(*mop)

        uevol = scipy.linalg.expm(-1j * env_qs.hmat * t)

        psi_in = u10 * env_qs.psi0
        psi = m10 * uevol * psi_in

        pvec = abs(np.array(psi.T)[0, :]) ** 2
        prob_samples[i, :] = pvec

    return prob_samples


# Calculating the rabi oscillations from the Hamitlonian directly
def rabi_oscillations_ham(moset, prepset, J_num, uvec, mvec, tvec):
    """
    Return p_um(t) = smcg model probability of different outcomes |0>, |1>, |2>, |3> for
    preparation u and measurement m at time t. Use general expressions for hamiltonian
    for evaluation.
    u in [0,1]
    m in [0,1,2]
    """
    mn = Nature(J_num)
    prob_samples = np.zeros((mvec.size, 4))

    for i in range(0, mvec.size):
        m = mvec[i]
        u = uvec[i]
        t = tvec[i]

        # Note the difference in how mop is defined compared to TwoQubitCoupling or CRModel cases
        uop = prepset[u]
        mop = moset[m]

        u10 = mn.kron(*uop)
        m10 = mn.kron(*mop)

        uevol = scipy.linalg.expm(-1j * mn.hmat * t)

        psi_in = u10 * mn.psi0
        psi = m10 * uevol * psi_in

        pvec = abs(np.array(psi.T)[0, :]) ** 2
        prob_samples[i, :] = pvec

    return 2*(prob_samples[:, 0] + prob_samples[:, 2]) - 1


# There are so many repetitions of this plotting routine ... really need to fix this!
def plot_rabi_oscillations(pvec_data, time_stamps, J_num=None, param_array=None,
                           FLAG_ibmq_boel=False, figsize_plot=(10,7),
                           FLAG_save=False, save_filename='llong_rabi_oscillations.png'):
    """
    This function plots the data with a lot of processing as required

    :param J_num:
    :param data:
    :param FLAG_readout_noise:
    :return:
    """
    # Calculate model probabilities
    if not (J_num is None):
        param_num = quantum_device_models.transform_parameters(J_num)
        teff = quantum_device_models.data_driven_teff_noise_model(param_num, FLAG_ibmq_boel=FLAG_ibmq_boel)

        # Get a more dense array of time_stamps
        DT = time_stamps[-1] - time_stamps[0]
        n_t = int(DT/(time_stamps[1] - time_stamps[0]))
        time_stamps_model = np.linspace(time_stamps[0], time_stamps[-1], n_t)
        # pvec_model = model_probabilities_analytic(time_stamps_model, teff=teff, J_num=J_num)

    if not (param_array is None):
        # Get a more dense array of time_stamps
        DT = time_stamps[-1] - time_stamps[0]
        n_t = int(DT / (time_stamps[1] - time_stamps[0]))
        time_stamps_model = np.linspace(time_stamps[0], time_stamps[-1], n_t)

        teff = quantum_device_models.data_driven_teff_noise_model(param_array, FLAG_ibmq_boel=FLAG_ibmq_boel)
        # pvec_model = model_probabilities_analytic(time_stamps_model, teff=teff, param_array=param_array)

    fig = plt.figure(3, figsize=figsize_plot)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    # plt.rcParams['figure.constrained_layout.use'] = True

    plt.subplot(311)
    plt.tick_params(axis='both', direction='in', labelbottom=False)
    if not (J_num is None) or not (param_array is None):
        plt.plot(time_stamps_model, pvec_model[0, :], '-b', label='model')
        plt.plot(time_stamps_model, pvec_model[1, :], '-r')

    plt.plot(time_stamps, pvec_data[0, :], 'ob', label='data')
    plt.plot(time_stamps, pvec_data[1, :], '^r')
    plt.ylabel(r"$\langle X \rangle$")
    plt.ylim((-1.2, 1.2))
    # plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.yticks(np.arange(-1, 1.5, step=0.5))
    # plt.gca().set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])

    plt.subplot(312)
    plt.tick_params(axis='both', direction='in', labelbottom=False)
    if not (J_num is None) or not (param_array is None):
        plt.plot(time_stamps_model, pvec_model[2, :], '-b', label='model')
        plt.plot(time_stamps_model, pvec_model[3, :], '-r')

    plt.plot(time_stamps, pvec_data[2, :], 'ob', label='data')
    plt.plot(time_stamps, pvec_data[3, :], '^r')
    plt.ylabel(r"$\langle Y \rangle$")
    plt.ylim((-1.2, 1.2))
    plt.yticks(np.arange(-1, 1.5, step=0.5))

    plt.subplot(313)
    plt.tick_params(axis='both', direction='in')
    if not (J_num is None) or not (param_array is None):
        L1 = plt.plot(time_stamps_model, pvec_model[4, :], '-b', label='model')
        L2 = plt.plot(time_stamps_model, pvec_model[5, :], '-r')

    L3 = plt.plot(time_stamps, pvec_data[4, :], 'ob', label='data')
    L4 = plt.plot(time_stamps, pvec_data[5, :], '^r')
    plt.xlabel("t (s)")
    plt.ylabel(r"$\langle Z \rangle$")
    plt.ylim((-1.2, 1.2))
    plt.yticks(np.arange(-1, 1.5, step=0.5))

    # Legend properties
    if not (J_num is None) or not (param_array is None):
        # Refs:
        # 1. https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib
        # 2. Why using [0] for the line objects? https://discourse.julialang.org/t/userwarning-legend-does-not-support-line2d-object/970/3
        labels_fig = (r'Model (Control in $|0\rangle$)', r'Model (Control in $|1\rangle$)',
                      r'Data (Control in $|0\rangle$)', r'Data (Control in $|1\rangle$)')

        fig.legend((L1[0], L2[0], L3[0], L4[0]), labels_fig, bbox_to_anchor=(0., -0.1, 1., 0.02),
                   loc='lower center', ncol=2, labelspacing=0., borderaxespad=0., fontsize=20)
    else:
        labels_fig = (r'Control in $|0\rangle$', r'Control in $|1\rangle$')

        fig.legend((L3[0], L4[0]), labels_fig, bbox_to_anchor=(0., -0.1, 1., 0.02),
                   loc='lower center', ncol=2, labelspacing=0., fontsize=20)

    plt.tight_layout()
    if FLAG_save is True:
        plt.savefig(save_filename, bbox_inches='tight')

    plt.show()


def compare_prediction_data(data, J_num=None, param_array=None, teff=np.array([0,0]),
                            A_cr=None, env_cr=None,
                            FLAG_return = False,
                            FLAG_readout_noise = True, do_plot=True, FLAG_save = False, save_filename='comparison1.eps',
                            figsize_plot=(10,7)):
    '''
    This function compares model prediction against data
    assuming fits

    Returns plots of model and data trends

    TO DO: Add options for the plotting rather than separate fields to be filled
    '''
    if 'device' in data.keys():
        if data['device'] == 'ibmq_boeblingen':
            if 'n_time_stamps' in data.keys():
                time_stamps = data['time_stamps'][0:int(data['n_time_stamps'])]
            else:
                time_stamps = data['time_stamps'][0:81]
    else:
        time_stamps = data['time_stamps'][0:-1:6]

    # Calculate model probabilities
    # if not(J_num is None):
    #     pvec_model = model_probabilities_analytic(time_stamps, teff=teff, J_num=J_num)
    #
    # if not(param_array is None):
    #     pvec_model = model_probabilities_analytic(time_stamps, teff=teff, param_array=param_array)

    if A_cr is None:
        pvec_data = process_data.prob_samples(data)
    else:
        pvec_data = A_cr.rabi_data(env_cr)

    if FLAG_readout_noise:
        if data['FLAG_classification'] is True:
            r0, r1 = data['misclassif_error']
            pvec_data = pvec_data/(1.0 - r0 - r1)

    #import pdb; pdb.set_trace()

    if do_plot:
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
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        plt.rcParams['text.usetex'] = True

        #plt.rcParams['figure.constrained_layout.use'] = True

        plt.subplot(311)
        plt.tick_params(axis='both', direction='in', labelbottom=False)
        if not(J_num is None) or not(param_array is None):
            plt.plot(time_stamps, pvec_model[0, :], '-b', label='model')
            plt.plot(time_stamps, pvec_model[1, :], '-r')

        plt.plot(time_stamps, pvec_data[0, :], 'ob', label='data')
        plt.plot(time_stamps, pvec_data[1, :], '^r')
        plt.ylabel(r"$\langle X \rangle$")
        plt.ylim((-1.2,1.2))
        #plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        plt.yticks(np.arange(-1, 1.5, step=0.5))
        #plt.gca().set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])

        plt.subplot(312)
        plt.tick_params(axis='both', direction='in', labelbottom=False)
        if not (J_num is None) or not (param_array is None):
            plt.plot(time_stamps, pvec_model[2, :], '-b', label='model')
            plt.plot(time_stamps, pvec_model[3, :], '-r')

        plt.plot(time_stamps, pvec_data[2, :], 'ob', label='data')
        plt.plot(time_stamps, pvec_data[3, :], '^r')
        plt.ylabel(r"$\langle Y \rangle$")
        plt.ylim((-1.2,1.2))
        plt.yticks(np.arange(-1, 1.5, step=0.5))

        plt.subplot(313)
        plt.tick_params(axis='both', direction='in')
        if not (J_num is None) or not (param_array is None):
            L1 = plt.plot(time_stamps, pvec_model[4, :], '-b', label='model')
            L2 = plt.plot(time_stamps, pvec_model[5, :], '-r')

        L3 = plt.plot(time_stamps, pvec_data[4, :], 'ob', label='data')
        L4 = plt.plot(time_stamps, pvec_data[5, :], '^r')
        plt.xlabel("t (s)")
        plt.ylabel(r"$\langle Z \rangle$")
        plt.ylim((-1.2,1.2))
        plt.yticks(np.arange(-1, 1.5, step=0.5))

        # Legend properties
        if not (J_num is None) or not (param_array is None):
            # Refs:
            # 1. https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib
            # 2. Why using [0] for the line objects? https://discourse.julialang.org/t/userwarning-legend-does-not-support-line2d-object/970/3
            labels_fig = (r'Model (Control in $|0\rangle$)', r'Model (Control in $|1\rangle$)',
                          r'Data (Control in $|0\rangle$)', r'Data (Control in $|1\rangle$)')

            fig.legend((L1[0], L2[0], L3[0], L4[0]), labels_fig, bbox_to_anchor=(0., -0.1, 1., 0.02),
                       loc='lower center', ncol=2, labelspacing=0., borderaxespad=0., fontsize=20)
        else:
            labels_fig = (r'Control in $|0\rangle$', r'Control in $|1\rangle$')

            fig.legend((L3[0], L4[0]), labels_fig, bbox_to_anchor=(0., -0.1, 1., 0.02),
                       loc='lower center', ncol=2, labelspacing=0., fontsize=20)

        plt.tight_layout()
        if FLAG_save is True:
            plt.savefig(save_filename, bbox_inches='tight', dpi=600)

        plt.show()

    if FLAG_return is True:
        if not (J_num is None) or not (param_array is None):
            return time_stamps, pvec_data, pvec_model
        else:
            return time_stamps, pvec_data

