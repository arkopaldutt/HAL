import numpy as np
import matplotlib.pyplot as plt


# Going back and forth between the different parameterizations
def transform_parameters(J_num):
    Jix = J_num[0]
    Jiy = J_num[1]
    Jiz = J_num[2]
    Jzx = J_num[3]
    Jzy = J_num[4]
    Jzz = J_num[5]

    a0 = Jiz + Jzz
    a1 = Jiz - Jzz
    beta0 = (Jix + Jzx) + 1j * (Jiy + Jzy)
    beta1 = (Jix - Jzx) + 1j * (Jiy - Jzy)

    omega0 = np.sqrt(a0 ** 2 + np.abs(beta0) ** 2)
    omega1 = np.sqrt(a1 ** 2 + np.abs(beta1) ** 2)
    delta0 = np.arcsin(a0 / omega0)
    delta1 = np.arcsin(a1 / omega1)
    phi0 = np.angle(beta0)
    phi1 = np.angle(beta1)

    param_array = np.array([omega0, delta0, phi0, omega1, delta1, phi1])

    return param_array


def transform_theta_to_J(theta_num):
    omega0, delta0, phi0, omega1, delta1, phi1 = theta_num

    a0 = omega0 * np.cos(delta0) * np.exp(1j * phi0) + omega1 * np.cos(delta1) * np.exp(1j * phi1)
    a1 = omega0 * np.cos(delta0) * np.exp(1j * phi0) - omega1 * np.cos(delta1) * np.exp(1j * phi1)

    b0 = omega0 * np.sin(delta0) + omega1 * np.sin(delta1)
    b1 = omega0 * np.sin(delta0) - omega1 * np.sin(delta1)

    Jix = (1 / 2) * np.real(a0)
    Jiy = (1 / 2) * np.imag(a0)
    Jiz = (1 / 2) * b0

    Jzx = (1 / 2) * np.real(a1)
    Jzy = (1 / 2) * np.imag(a1)
    Jzz = (1 / 2) * b1

    J_num = np.array([Jix, Jiy, Jiz, Jzx, Jzy, Jzz])

    return J_num


# Imperfect-Pulse Shaping Nonideality or Control Noise Models
def data_driven_teff_noise_model(param_num, FLAG_ibmq_boel=False, device_id='A'):
    delta_t_imperfect_pulse = lambda omega, coeff0, coeff1: coeff0 / (omega + coeff1 * omega ** 2)

    omega0 = param_num[0]
    omega1 = param_num[3]

    # Calculate teff0 and teff1 using the curve fits
    if FLAG_ibmq_boel:
        #coeff0 = 6.39193733
        #coeff1 = 5.45719797e-09
        coeff0 = 6.27739558
        coeff1 = 1.50856579e-09
    else:
        if device_id == 'A':
            coeff0 = 6.24185261
            coeff1 = 5.85174452e-09
        elif device_id == 'C':
            coeff0 = 6.28120836
            coeff1 = 9.10033846e-09
        else:
            coeff0 = 3.1458
            coeff1 = 1.423e-8


    teff0 = delta_t_imperfect_pulse(omega0, coeff0, coeff1)
    teff1 = delta_t_imperfect_pulse(omega1, coeff0, coeff1)

    return np.array([teff0,teff1])


def grad_data_driven_teff_noise_model(param_num, FLAG_ibmq_boel=False):
    # Function to calculate d( \omega \Delta t_eff)/d \omega
    grad_delta_t_imperfect_pulse = lambda omega, coeff0, coeff1: - coeff0 * coeff1 / ((1 + coeff1 * omega)**2)

    omega0 = param_num[0]
    omega1 = param_num[3]

    # Calculate teff0 and teff1 using the curve fits
    if FLAG_ibmq_boel:
        # coeff0 = 6.39193733
        # coeff1 = 5.45719797e-09
        coeff0 = 6.27739558
        coeff1 = 1.50856579e-09
    else:
        coeff0 = 3.1458
        coeff1 = 1.423e-8

    grad_teff0 = grad_delta_t_imperfect_pulse(omega0, coeff0, coeff1)
    grad_teff1 = grad_delta_t_imperfect_pulse(omega1, coeff0, coeff1)

    return np.array([grad_teff0, grad_teff1])


# Decoherence Models
def decoherence_single_qubit_model(t, t1, t2, n_qubits=1, n_target=1):
    """
    Considering A = 1/3 exp(-t/t1) + 2/3 exp(-t/t2)

    returns [A, 1/2**n_qubits * (1-A)]

    Assuming for one target qubit, the resulting probability is then A*q + (1/2**n_qubits)*(1-A)
    """
    damping_factor = (1/3)*np.exp(-t/t1) + (2/3)*np.exp(-t/t2)

    return np.array([damping_factor, (1/(2**n_qubits))*(1-damping_factor)])


def decoherence_two_qubit_model(t, t1_array, t2_array, n_qubits=2, n_target=1):
    """
    Considering A = 1/3 exp(-t/t1) + 2/3 exp(-t/t2)

    returns [A, 1/2**n_qubits * (1-A)]

    Assuming for one target qubit, the resulting probability is then A*q + (1/2**n_qubits)*(1-A)
    """
    ga1 = 1 - np.exp(-t/(2*t1_array[0]))
    ga2 = 1 - np.exp(-t/(2*t1_array[1]))

    inv_tp_array = [ 1/t2_array[ind] - 1/(2*t1_array[ind]) for ind in range(2)]

    gp1 = 1 - np.exp(-t * inv_tp_array[0])
    gp2 = 1 - np.exp(-t * inv_tp_array[1])

    d1 = (ga1**2)*(3*ga2**2 + 4*ga2*(gp2-2) - 4*gp2 + 7)
    d2 = 4*ga1*(gp1-2)*(ga2**2 + ga2*(gp2-2) - gp2 + 2)
    d3 = (ga2**2)*(7-4*gp1) - 4*ga2*(gp1-2)*(gp2-2) + 4*gp1*gp2 - 8*gp1 - 8*gp2

    damping_factor = 1 + (1/15)*(d1 + d2 + d3)

    return np.array([damping_factor, (1/(2**n_qubits))*(1-damping_factor)])


def decoherence_multiqubit_model(t, t1_array, t2_array, n_qubits=2, n_target=1):
    """
    https: // qiskit.org / documentation / _modules / qiskit / ignis / verification / randomized_benchmarking / rb_utils.html  # coherence_limit
    """


# Leakage Error Models -- only single qubit
def erasure_error_model(pvec_target, leakage_rate):
    """
    We note that the input pvec is a probability vector over [|00>, |01>, |10>, |11>]
    """

    L1 = leakage_rate[0]
    pvec_leaked = np.copy(pvec_target)

    # pvec_leaked[0] = ((1-L1)**2)*pvec[0]
    # pvec_leaked[1] = (1-L1)*pvec[1] + L1*pvec[0]
    # pvec_leaked[2] = (1-L1)*pvec[2] + L1*pvec[0]
    # pvec_leaked[3] = ((1-L1)**2)*pvec[3] + (L1**2)*pvec[0] + L1*pvec[1] + L1*pvec[2]
    pvec_leaked[0] = (1-L1)*pvec_target[0]
    pvec_leaked[1] = pvec_target[1] + L1*pvec_target[0]

    return pvec_leaked


# Leakage Error Models -- Depolarizing Leakage Extensions (DLE)
def DLE_linear(pvec_target, leakage_rate, t, mu_T):
    """
    We note that the input pvec is a probability vector over [|00>, |01>, |10>, |11>]
    """

    L1 = leakage_rate[0]*(np.exp(-t/mu_T))
    pvec_leaked = np.copy(pvec_target)

    # pvec_leaked[0] = ((1-L1)**2)*pvec[0]
    # pvec_leaked[1] = (1-L1)*pvec[1] + L1*pvec[0]
    # pvec_leaked[2] = (1-L1)*pvec[2] + L1*pvec[0]
    # pvec_leaked[3] = ((1-L1)**2)*pvec[3] + (L1**2)*pvec[0] + L1*pvec[1] + L1*pvec[2]
    pvec_leaked[0] = (1 - L1) * pvec_target[0]
    pvec_leaked[1] = pvec_target[1] + L1 * pvec_target[0]

    return pvec_leaked


# Leakage Error Models -- Depolarizing Leakage Extensions (DLE)
def DLE_quadratic(pvec_target, leakage_rate, t, mu_T):
    """
    We note that the input pvec is a probability vector over [|00>, |01>, |10>, |11>]
    """

    L1 = leakage_rate[0]*(np.exp(-t**2/mu_T))
    pvec_leaked = np.copy(pvec_target)

    # pvec_leaked[0] = ((1-L1)**2)*pvec[0]
    # pvec_leaked[1] = (1-L1)*pvec[1] + L1*pvec[0]
    # pvec_leaked[2] = (1-L1)*pvec[2] + L1*pvec[0]
    # pvec_leaked[3] = ((1-L1)**2)*pvec[3] + (L1**2)*pvec[0] + L1*pvec[1] + L1*pvec[2]
    pvec_leaked[0] = (1 - L1) * pvec_target[0]
    pvec_leaked[1] = pvec_target[1] + L1 * pvec_target[0]

    return pvec_leaked


# Overall Model of Quantum System
class SystemModel(object):
    """
    Describes the model of the system as known to the Learner
    In other words, represents what the Learner thinks Nature is

    A lot of the methods thus intersects with the classes ActionSpace and Nature

    TO DO: May want to make these functions more independent of each other
    """

    def __init__(self, J, xi_J, xi_param=None,
                 noise=None,
                 FLAG_readout_noise = False,
                 FLAG_control_noise = False,
                 FLAG_decoherence = False,
                 FLAG_leakage = False,
                 FLAG_single_qubit_noise_model = False):
        """
        Just one thing to note is that the SystemModel can and will be wrong compared to the oracle!
        So we could assume there are no noise sources here even though in real life, there is!

        In our workflow, the noise is learnt offline and hence will be filled in with the true values of
        the Nature more or less but this is not necessary usually
        """
        # As obtained usually from the estimator or prior information
        Jix, Jiy, Jiz, Jzx, Jzy, Jzz = J

        self.J = J
        self.xi_J = xi_J
        self.J_nd = J/xi_J

        self.Jix = Jix
        self.Jiy = Jiy
        self.Jiz = Jiz
        self.Jzx = Jzx
        self.Jzy = Jzy
        self.Jzz = Jzz

        self.Jix_nd = self.J_nd[0]
        self.Jiy_nd = self.J_nd[1]
        self.Jiz_nd = self.J_nd[2]
        self.Jzx_nd = self.J_nd[3]
        self.Jzy_nd = self.J_nd[4]
        self.Jzz_nd = self.J_nd[5]

        self.param_array = transform_parameters(self.J)

        # A particular choice for the time being that might need to be changed later
        if xi_param is None:
            self.xi_param = np.array([np.amax(xi_J), 1, 1, np.amax(xi_J), 1, 1])
        else:
            self.xi_param = xi_param

        self.param_nd = self.param_array/self.xi_param

        self.jacobian_J_param = self.transform_param_sets()

        # Noise
        self.FLAG_readout_noise = FLAG_readout_noise
        self.FLAG_control_noise = FLAG_control_noise
        self.FLAG_decoherence_noise = FLAG_decoherence
        self.FLAG_leakage_error = FLAG_leakage

        if noise is not None:
            if FLAG_readout_noise is True:
                # Note that these are slightly different from what we have in the case of the Nature object
                self.readout_noise = noise['readout']
            else:
                self.readout_noise = np.array([0.0, 0.0])

            if FLAG_control_noise is True:
                # function handles (partial) such that we can directly apply these functions to parameters
                self.control_noise_model = noise['control_noise']
                self.grad_control_noise_model = noise['grad_control_noise']
            else:
                self.control_noise_model = None
                self.grad_control_noise_model = None

            if FLAG_decoherence is True:
                self.T1 = noise['T1']   # expect array of form [control T1, target T1]
                self.T2 = noise['T2']   # expect array of form [control T2, target T2]

                # decoherence model for a single qubit at the moment using average T1 and T2 of the 2 qubits
                self.decoherence_noise_model = noise['decoherence']
                self.FLAG_single_qubit_noise_model = FLAG_single_qubit_noise_model

            if FLAG_leakage is True:
                self.leakage_rate = noise['leakage_rate'] # expect array of form [leakage L1, seepage L2]

                self.leakage_error_model = noise['leakage_model']
                self.mu_T = noise['mu_T']

    def print_info(self):
        print('Quantum Device Model Info:')
        print('Noise Sources:')
        print('Readout Noise: FLAG=%r, Value=%s' %(self.FLAG_readout_noise, self.readout_noise))
        print('Control Noise: FLAG=%r' % self.FLAG_control_noise)
        print('Decoherence: FLAG=%r' % self.FLAG_decoherence_noise)
        print('Leakage Error: FLAG=%r' % self.FLAG_leakage_error)

    def update(self, J, xi_J=None, xi_param=None):
        # Usually after a learning step and assuming xi_J remains constant
        Jix, Jiy, Jiz, Jzx, Jzy, Jzz = J

        self.J = J

        if xi_J is not None:
            self.xi_J = xi_J

        self.J_nd = J / self.xi_J

        self.Jix = Jix
        self.Jiy = Jiy
        self.Jiz = Jiz
        self.Jzx = Jzx
        self.Jzy = Jzy
        self.Jzz = Jzz

        self.Jix_nd = self.J_nd[0]
        self.Jiy_nd = self.J_nd[1]
        self.Jiz_nd = self.J_nd[2]
        self.Jzx_nd = self.J_nd[3]
        self.Jzy_nd = self.J_nd[4]
        self.Jzz_nd = self.J_nd[5]

        self.param_array = transform_parameters(self.J)

        if xi_param is not None:
            self.xi_param = xi_param

        self.param_nd = self.param_array / self.xi_param

        self.jacobian_J_param = self.transform_param_sets()

    def model_probabilities(self, n_config, t, FLAG_noise=False):
        """
        n_config = what is the combination of M and U being considered? n_config = 2*m + u typically
        FLAG_noise indicates whether the Learner wants the noise to be considered even if present in the SystemModel

        returns p(observation|query), observation has gone through noisy system evolution and noisy measurement channel
        """
        omega0, delta0, phi0, omega1, delta1, phi1 = self.param_array
        duration_pulse = np.copy(t)

        if FLAG_noise and self.FLAG_control_noise:
            teff0, teff1 = self.control_noise_model(self.param_array)

            if n_config % 2 == 0:
                t = t + teff0
            else:
                t = t + teff1

        pvec = np.zeros(4)
        if (n_config == 0):
            pvec[0] = 0.5 * ((np.cos(omega0 * t) + np.sin(phi0) * np.cos(delta0) * np.sin(omega0 * t)) ** 2 + (
                    np.sin(delta0) * np.sin(omega0 * t) + np.cos(phi0) * np.cos(delta0) * np.sin(omega0 * t)) ** 2)
            pvec[1] = 1 - pvec[0]
            pvec[2] = 0
            pvec[3] = 0
        elif (n_config == 1):
            pvec[0] = 0
            pvec[1] = 0
            pvec[2] = 0.5 * ((np.cos(omega1 * t) + np.sin(phi1) * np.cos(delta1) * np.sin(omega1 * t)) ** 2 + (
                    np.sin(delta1) * np.sin(omega1 * t) + np.cos(phi1) * np.cos(delta1) * np.sin(omega1 * t)) ** 2)
            pvec[3] = 1 - pvec[2]
        elif (n_config == 2):
            pvec[0] = 0.5 * ((np.cos(omega0 * t) - np.cos(phi0) * np.cos(delta0) * np.sin(omega0 * t)) ** 2 + (
                    np.sin(delta0) * np.sin(omega0 * t) + np.sin(phi0) * np.cos(delta0) * np.sin(omega0 * t)) ** 2)
            pvec[1] = 1 - pvec[0]
            pvec[2] = 0
            pvec[3] = 0
        elif (n_config == 3):
            pvec[0] = 0
            pvec[1] = 0
            pvec[2] = 0.5 * ((np.cos(omega1 * t) - np.cos(phi1) * np.cos(delta1) * np.sin(omega1 * t)) ** 2 + (
                    np.sin(delta1) * np.sin(omega1 * t) + np.sin(phi1) * np.cos(delta1) * np.sin(omega1 * t)) ** 2)
            pvec[3] = 1 - pvec[2]
        elif (n_config == 4):
            pvec[0] = 1 - (np.cos(delta0) ** 2) * (np.sin(omega0 * t) ** 2)
            pvec[1] = (np.cos(delta0) ** 2) * (np.sin(omega0 * t) ** 2)
            pvec[2] = 0
            pvec[3] = 0
        elif (n_config == 5):
            pvec[0] = 0
            pvec[1] = 0
            pvec[2] = 1 - (np.cos(delta1) ** 2) * (np.sin(omega1 * t) ** 2)
            pvec[3] = (np.cos(delta1) ** 2) * (np.sin(omega1 * t) ** 2)

        if FLAG_noise and self.FLAG_decoherence_noise:
            damping_array = self.decoherence_noise_model(duration_pulse)

            if self.FLAG_single_qubit_noise_model:
                pvec[0] = damping_array[0]*(pvec[0] + pvec[2]) + damping_array[1]
                pvec[1] = damping_array[0]*(pvec[1] + pvec[3]) + damping_array[1]
                pvec[2] = 0
                pvec[3] = 0
                return pvec
            else:
                return damping_array[0]*pvec + damping_array[1]
        else:
            return pvec

    def target_probability(self, n_config, t, FLAG_noise=False):
        # Get probability of target qubit
        pvec = self.model_probabilities(n_config, t, FLAG_noise=FLAG_noise)

        # No further noise case
        pvec_target = [pvec[0] + pvec[2], pvec[1] + pvec[3]] # [p0_ideal, p1_ideal]

        # Accounting for leakage errors if any
        if FLAG_noise and self.FLAG_leakage_error:
            if self.leakage_error_model == 'erasure':
                #print('Erasure')
                pvec_target = erasure_error_model(pvec_target, self.leakage_rate)
            elif self.leakage_error_model == 'dle_linear':
                #print('DLE -- Linear')
                pvec_target = DLE_linear(pvec_target, self.leakage_rate, t, self.mu_T)
            elif self.leakage_error_model == 'dle_quadratic':
                #print('DLE -- Quadratic')
                pvec_target = DLE_quadratic(pvec_target, self.leakage_rate, t, self.mu_T)
            else:
                pvec_target = self.leakage_error_model(pvec_target, self.leakage_rate)

        # Assuming that the target qubit is measured as usual
        if FLAG_noise and self.FLAG_readout_noise:
            r0, r1 = self.readout_noise
            p0 = (1 - r0) * pvec_target[0] + r1 * pvec_target[1]
        else:
            p0 = pvec_target[0]

        return p0

    def rabi_oscillations(self, n_config, t, FLAG_noise=False):
        # Get probability of target qubit
        p0 = self.target_probability(n_config, t, FLAG_noise=FLAG_noise)

        return 2*p0 - 1

    def analytical_rabi_oscillations(self, n_config, t, FLAG_noise=False):
        omega0, delta0, phi0, omega1, delta1, phi1 = self.param_array

        alpha0 = np.angle( -np.sin(delta0) * np.cos(phi0) - 1j * np.sin(phi0) )
        alpha1 = np.angle( -np.sin(delta1) * np.cos(phi1) - 1j * np.sin(phi1) )
        gamma0 = np.angle( -np.sin(delta0) * np.sin(phi0) + 1j * np.cos(phi0) )
        gamma1 = np.angle( -np.sin(delta1) * np.sin(phi1) + 1j * np.cos(phi1) )

        p_rabi = 0.0
        if n_config == 0:
            p_rabi = np.sin(delta0)*np.cos(delta0)*np.cos(phi0) + \
                     np.cos(delta0)*np.sqrt(1 - (np.cos(delta0)**2)*(np.cos(phi0)**2))*np.cos(2*omega0*t + alpha0)

        elif n_config == 1:
            p_rabi = np.sin(delta1)*np.cos(delta1) * np.cos(phi1) + \
                     np.cos(delta1) * np.sqrt(1 - (np.cos(delta1)**2) * (np.cos(phi1)**2))*np.cos(2*omega1*t + alpha1)

        elif n_config == 2:
            p_rabi = np.sin(delta0)*np.cos(delta0)*np.sin(phi0) + \
                     np.cos(delta0)*np.sqrt(1 - (np.cos(delta0)**2)*(np.sin(phi0)**2))*np.cos(2*omega0*t + gamma0)
        elif n_config == 3:
            p_rabi = np.sin(delta1)*np.cos(delta1)*np.sin(phi1) + \
                     np.cos(delta1)*np.sqrt(1 - (np.cos(delta1)**2)*(np.sin(phi1)**2))*np.cos(2*omega1*t + gamma1)
        elif n_config == 4:
            p_rabi = 1 - 2*(np.cos(delta0) ** 2) * (np.sin(omega0 * t) ** 2)
        elif n_config == 5:
            p_rabi = 1 - 2*(np.cos(delta1) ** 2) * (np.sin(omega1 * t) ** 2)

        return p_rabi

    def zeros_rabi_oscillations(self, n_config, lower_bound_t=1e-7, FLAG_noise=False):
        u_op = np.mod(n_config,2)
        m_op = int(round((n_config - u_op) / 2, 0))

        omega_array = np.array([self.param_array[0], self.param_array[3]])
        delta_array = np.array([self.param_array[1], self.param_array[4]])
        phi_array = np.array([self.param_array[2], self.param_array[5]])

        omega = omega_array[u_op]
        delta = delta_array[u_op]
        phi = phi_array[u_op]

        alpha = np.angle(-np.sin(delta) * np.cos(phi) - 1j * np.sin(phi))
        gamma = np.angle(-np.sin(delta) * np.sin(phi) + 1j * np.cos(phi))

        zero_rabi = 0.0
        if m_op == 0:
            num = -np.sin(delta) * np.cos(phi)
            denom = np.sqrt(1 - (np.cos(delta) ** 2) * (np.cos(phi) ** 2))
            zero_rabi = (np.arccos(num / denom) - alpha) / (2 * omega)

        elif m_op == 1:
            num = -np.sin(delta) * np.sin(phi)
            denom = np.sqrt(1 - (np.cos(delta) ** 2) * (np.sin(phi) ** 2))
            zero_rabi = (np.arccos(num / denom) - gamma) / (2 * omega)

        elif m_op == 2:
            zero_rabi = np.arccos(-np.tan(delta)**2)/ (2*omega)

        # Get the appropriate zeros
        zero_shift = np.pi/(2*omega)
        while zero_rabi < lower_bound_t:
            zero_rabi += zero_shift

        while zero_rabi > lower_bound_t:
            zero_rabi -= zero_shift

        # This ensures the returned zero_rabi is less than the lower_bound_t but + zero_shift > lower_bound_t
        return zero_rabi

    def actions_zeros_rabi_oscillations(self, config_array, k_array, xi_t=1e-7, dt=6.8323e-9, FLAG_noise=False):
        """
        configs_array has elements of (2*m + u) values. Used if we don't want all the 6 possible configurations to be
        used
        """
        omega = np.array([self.param_array[0], self.param_array[3]])
        N_configs = len(config_array)

        actions_rabi = []
        for ind_config in config_array:
            # Get (M,U) corresponding to this configuration
            # tuple_temp = np.base_repr(ind_config, base=3, padding=0)
            # if ind_config == 0:
            #     tuple_temp = np.base_repr(ind_config, base=3, padding=2)
            # elif len(tuple_temp) == 1:
            #     tuple_temp = np.base_repr(ind_config, base=3, padding=1)
            # if len(tuple_temp) > 2:
            #     print('We got problems with the number of configurations being used here!')

            # m_temp = int(tuple_temp[0])
            # u_temp = int(tuple_temp[1])
            u_temp = np.mod(ind_config, 2)
            m_temp = int(round((ind_config - u_temp)/2, 0))

            # Get the zero in the rabi oscillation corresponding to this configuration
            zero_rabi_temp = self.zeros_rabi_oscillations(ind_config)

            for k in k_array:
                tnd_temp = (zero_rabi_temp + k*np.pi/(2*omega[u_temp]))/xi_t
                actions_rabi.append([m_temp, u_temp, tnd_temp])
                actions_rabi.append([m_temp, u_temp, tnd_temp - dt/xi_t])
                actions_rabi.append([m_temp, u_temp, tnd_temp + dt/xi_t])
                actions_rabi.append([m_temp, u_temp, tnd_temp - 2*dt/xi_t])
                actions_rabi.append([m_temp, u_temp, tnd_temp + 2*dt/xi_t])

        return actions_rabi

    def actions_rabi_oscillations(self, config_array, k_array, xi_t=1e-7, dt=6.8323e-9, factor_max=0.1, FLAG_noise=False):
        """
        configs_array has elements of (2*m + u) values. Used if we don't want all the 6 possible configurations to be
        used

        zero crossings and max amplitude values
        """
        omega = np.array([self.param_array[0], self.param_array[3]])
        N_configs = len(config_array)

        actions_rabi = []
        for ind_config in config_array:
            # Get (M,U) corresponding to this configuration
            # tuple_temp = np.base_repr(ind_config, base=3, padding=0)
            # if ind_config == 0:
            #     tuple_temp = np.base_repr(ind_config, base=3, padding=2)
            # elif len(tuple_temp) == 1:
            #     tuple_temp = np.base_repr(ind_config, base=3, padding=1)
            # if len(tuple_temp) > 2:
            #     print('We got problems with the number of configurations being used here!')

            # m_temp = int(tuple_temp[0])
            # u_temp = int(tuple_temp[1])
            u_temp = np.mod(ind_config, 2)
            m_temp = int(round((ind_config - u_temp)/2, 0))

            # Get the zero in the rabi oscillation corresponding to this configuration
            zero_rabi_temp = self.zeros_rabi_oscillations(ind_config)

            for k in k_array:
                tnd_temp = (zero_rabi_temp + k*np.pi/(2*omega[u_temp]))/xi_t
                actions_rabi.append([m_temp, u_temp, tnd_temp])
                actions_rabi.append([m_temp, u_temp, tnd_temp - dt/xi_t])
                actions_rabi.append([m_temp, u_temp, tnd_temp + dt/xi_t])
                actions_rabi.append([m_temp, u_temp, tnd_temp - 2*dt/xi_t])
                actions_rabi.append([m_temp, u_temp, tnd_temp + 2*dt/xi_t])

            for k in k_array:
                tnd_temp = ( zero_rabi_temp/4 + k*np.pi/(4*omega[u_temp]) )/xi_t
                actions_rabi.append([m_temp, u_temp, tnd_temp])
                actions_rabi.append([m_temp, u_temp, tnd_temp - dt/xi_t])
                actions_rabi.append([m_temp, u_temp, tnd_temp + dt/xi_t])
                actions_rabi.append([m_temp, u_temp, tnd_temp - 2*dt/xi_t])
                actions_rabi.append([m_temp, u_temp, tnd_temp + 2*dt/xi_t])

        return actions_rabi

    def log_likelihood(self, n_config, t, FLAG_noise=False):
        """
        Returns log-likelihood $-\log( p_{\mathrm{y}|\mathrm{x}}(0|x) ) and assuming only target qubit is measured
        """
        # Get model probability and return log of that
        pvec = self.model_probabilities(n_config, t,  FLAG_noise=FLAG_noise)

        # Assuming that the target qubit is measured as usual
        if FLAG_noise and self.FLAG_readout_noise:
            r0, r1 = self.readout_noise
            p0 = (1 - r0) * (pvec[0] + pvec[2]) + r1 * (pvec[1] + pvec[3])
        else:
            p0 = pvec[0] + pvec[2]
            
        return -np.log(p0)

    def log_likelihood_loss(self, y, n_config, t, FLAG_noise=False):
        """
        Inputs:
        y - measurement
        Returns log-likelihood $-\log( p_{\mathrm{y}|\mathrm{x}}(0|x) ) and assuming only target qubit is measured
        """
        # Get model probability and return log of that
        pvec = self.model_probabilities(n_config, t, FLAG_noise=FLAG_noise)

        # Assuming that the target qubit is measured as usual
        p0 = pvec[0] + pvec[2]
        if FLAG_noise and self.FLAG_readout_noise:
            r0, r1 = self.readout_noise
        else:
            r0 = 0.0
            r1 = 0.0

        loss = (1 - y) * ((1 - r0) * p0 + r1 * (1 - p0)) + \
               y * ((1 - r1) * (1 - p0) + r0 * p0)  # log-likelihood

        return -np.log(loss)

    def jacobian_likelihood(self, n_config, t, FLAG_noise=False, type_param='J'):
        """
        Define function to calculate the score or the jacobian vector considering state 0
        $\frac{\partial}{\partial \theta} \log(p(y;\theta))$

        Note that grad_control_noise is user defined. So for instance if there are FLAGs as there are
        in the case of estimators.data_driven_model then make sure to include relevant value (bool in this case)
        when sending over the partial function to this class.
        """

        omega0, delta0, phi0, omega1, delta1, phi1 = self.param_array
        duration_pulse = np.copy(t)

        # Default value of d( \omega t + \omega \Delta t_eff)/d \omega
        darg_omega_t_domega = np.copy(t)

        if FLAG_noise and self.FLAG_control_noise:
            teff0, teff1 = self.control_noise_model(self.param_array)
            d_omega_teff0_d_omega, d_omega_teff1_d_omega = self.grad_control_noise_model(self.param_array)

            if n_config % 2 == 0:
                t = t + teff0
                darg_omega_t_domega += d_omega_teff0_d_omega
            else:
                t = t + teff1
                darg_omega_t_domega += d_omega_teff1_d_omega

        jacobian = np.zeros(6)
        if (n_config == 0):
            jacobian[0] = darg_omega_t_domega * (
                        np.sin(delta0) * np.cos(phi0) * np.sin(2 * omega0 * t) + np.sin(phi0) * np.cos(
                    2 * omega0 * t)) * np.cos(delta0)
            jacobian[1] = (np.sin(omega0 * t) * np.cos(phi0) * np.cos(2 * delta0) - np.sin(delta0) * np.sin(
                phi0) * np.cos(
                omega0 * t)) * np.sin(omega0 * t)
            jacobian[2] = (-np.sin(delta0) * np.sin(phi0) * np.sin(omega0 * t) + np.cos(phi0) * np.cos(
                omega0 * t)) * np.sin(omega0 * t) * np.cos(delta0)
        elif (n_config == 1):
            jacobian[3] = darg_omega_t_domega * (
                        np.sin(delta1) * np.cos(phi1) * np.sin(2 * omega1 * t) + np.sin(phi1) * np.cos(
                    2 * omega1 * t)) * np.cos(delta1)
            jacobian[4] = (np.sin(omega1 * t) * np.cos(phi1) * np.cos(2 * delta1) - np.sin(delta1) * np.sin(
                phi1) * np.cos(
                omega1 * t)) * np.sin(omega1 * t)
            jacobian[5] = (-np.sin(delta1) * np.sin(phi1) * np.sin(omega1 * t) + np.cos(phi1) * np.cos(
                omega1 * t)) * np.sin(omega1 * t) * np.cos(delta1)
        elif (n_config == 2):
            jacobian[0] = darg_omega_t_domega * (
                        np.sin(delta0) * np.sin(phi0) * np.sin(2 * omega0 * t) - np.cos(phi0) * np.cos(
                    2 * omega0 * t)) * np.cos(delta0)
            jacobian[1] = (np.sin(omega0 * t) * np.sin(phi0) * np.cos(2 * delta0) + np.sin(delta0) * np.cos(
                phi0) * np.cos(
                omega0 * t)) * np.sin(omega0 * t)
            jacobian[2] = (np.sin(delta0) * np.cos(phi0) * np.sin(omega0 * t) + np.sin(phi0) * np.cos(
                omega0 * t)) * np.sin(
                omega0 * t) * np.cos(delta0)
        elif (n_config == 3):
            jacobian[3] = darg_omega_t_domega * (
                        np.sin(delta1) * np.sin(phi1) * np.sin(2 * omega1 * t) - np.cos(phi1) * np.cos(
                    2 * omega1 * t)) * np.cos(delta1)
            jacobian[4] = (np.sin(omega1 * t) * np.sin(phi1) * np.cos(2 * delta1) + np.sin(delta1) * np.cos(
                phi1) * np.cos(
                omega1 * t)) * np.sin(omega1 * t)
            jacobian[5] = (np.sin(delta1) * np.cos(phi1) * np.sin(omega1 * t) + np.sin(phi1) * np.cos(
                omega1 * t)) * np.sin(
                omega1 * t) * np.cos(delta1)
        elif (n_config == 4):
            jacobian[0] = -darg_omega_t_domega * np.sin(2 * omega0 * t) * (np.cos(delta0) ** 2)
            jacobian[1] = np.sin(2 * delta0) * (np.sin(omega0 * t) ** 2)
        elif (n_config == 5):
            jacobian[3] = -darg_omega_t_domega * np.sin(2 * omega1 * t) * (np.cos(delta1) ** 2)
            jacobian[4] = np.sin(2 * delta1) * (np.sin(omega1 * t) ** 2)

        if FLAG_noise and self.FLAG_decoherence_noise:
            damping_array = self.decoherence_noise_model(duration_pulse)
            jacobian = damping_array[0]*jacobian

        if FLAG_noise and self.FLAG_leakage_error:
            if self.leakage_error_model == 'erasure':
                L1 = self.leakage_rate[0]
            elif self.leakage_error_model == 'dle_linear':
                L1 = self.leakage_rate[0]*np.exp(-t/self.mu_T)
            elif self.leakage_error_model == 'dle_quadratic':
                L1 = self.leakage_rate[0] * np.exp(-t**2/self.mu_T)

            jacobian = (1 - L1) * jacobian

        if FLAG_noise and self.FLAG_readout_noise:
            r0, r1 = self.readout_noise
            jacobian = (1 - r0 - r1) * jacobian

        if type_param == 'param':
            return jacobian
        elif type_param == 'J':
            return np.dot(self.jacobian_J_param, jacobian)
        else:
            raise RuntimeError('Wrong argument of type_param')

    def model_probability_target_qubit(self, n_config, t, FLAG_noise=False):
        pvec = self.model_probabilities(n_config, t, FLAG_noise=FLAG_noise)

        # Assuming that the target qubit is measured as usual
        if FLAG_noise and self.FLAG_readout_noise:
            r0, r1 = self.readout_noise
            p0 = (1 - r0) * (pvec[0] + pvec[2]) + r1 * (pvec[1] + pvec[3])
        else:
            p0 = pvec[0] + pvec[2]

        return p0

    def jacobian_log_likelihood(self, n_config, t, FLAG_noise=False, FLAG_normalization=False, type_param='J'):
        pvec = self.model_probabilities(n_config, t, FLAG_noise=FLAG_noise)
        jacobian = self.jacobian_likelihood(n_config, t, FLAG_noise=FLAG_noise, type_param=type_param)

        if FLAG_normalization:
            if type_param == 'J':
                jacobian = self.xi_J * jacobian
            elif type_param == 'param':
                jacobian = self.xi_param * jacobian
            else:
                raise RuntimeError('Wrong argument for type_param')

        # Assuming that the target qubit is measured as usual
        # if FLAG_noise and self.FLAG_readout_noise:
        #     r0, r1 = self.readout_noise
        #     p0 = (1 - r0) * (pvec[0] + pvec[2]) + r1 * (pvec[1] + pvec[3])
        # else:
        #     p0 = pvec[0] + pvec[2]
        p0 = self.target_probability(n_config, t, FLAG_noise=FLAG_noise)

        if n_config % 2 == 0:
            factor_temp = 1 / (p0 + 1e-10)
        else:
            factor_temp = 1 / ( (1-p0) + 1e-10)

        return factor_temp*jacobian

    def transform_param_sets(self):
        '''
        Defines the jacobian to go from (omega0, delta0, phi0, omega1, delta1, phi1) to (Jix, Jiy, Jiz, Jzx, Jzy, Jzz)
        '''
        Jix, Jiy, Jiz, Jzx, Jzy, Jzz = self.J

        # Derivatives wrt Jix
        dw0_dJix = (Jix + Jzx) / np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2)
        dw1_dJix = (Jix - Jzx) / np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2)
        dd0_dJix = -(Jix + Jzx) * (Jiz + Jzz) / (np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2) * (
                    (Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2))
        dd1_dJix = -(Jix - Jzx) * (Jiz - Jzz) / (np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2) * (
                    (Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2))
        dp0_dJix = -(Jiy + Jzy) / ((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2)
        dp1_dJix = (-Jiy + Jzy) / ((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2)

        # Derivatives wrt Jiy
        dw0_dJiy = (Jiy + Jzy) / np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2)
        dw1_dJiy = (Jiy - Jzy) / np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2)
        dd0_dJiy = -(Jiy + Jzy) * (Jiz + Jzz) / (np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2) * (
                    (Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2))
        dd1_dJiy = -(Jiy - Jzy) * (Jiz - Jzz) / (np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2) * (
                    (Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2))
        dp0_dJiy = (Jix + Jzx) / ((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2)
        dp1_dJiy = (Jix - Jzx) / ((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2)

        # Derivatives wrt Jiz
        dw0_dJiz = (Jiz + Jzz) / np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2)
        dw1_dJiz = (Jiz - Jzz) / np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2)
        dd0_dJiz = np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2) / (
                    (Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2)
        dd1_dJiz = np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2) / (
                    (Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2)
        dp0_dJiz = 0
        dp1_dJiz = 0

        # Derivatives wrt Jzx
        dw0_dJzx = (Jix + Jzx) / np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2)
        dw1_dJzx = (-Jix + Jzx) / np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2)
        dd0_dJzx = -(Jix + Jzx) * (Jiz + Jzz) / (np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2) * (
                    (Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2))
        dd1_dJzx = (Jix - Jzx) * (Jiz - Jzz) / (np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2) * (
                    (Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2))
        dp0_dJzx = -(Jiy + Jzy) / ((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2)
        dp1_dJzx = (Jiy - Jzy) / ((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2)

        # Derivatives wrt Jzy
        dw0_dJzy = (Jiy + Jzy) / np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2)
        dw1_dJzy = (-Jiy + Jzy) / np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2)
        dd0_dJzy = -(Jiy + Jzy) * (Jiz + Jzz) / (np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2) * (
                    (Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2))
        dd1_dJzy = (Jiy - Jzy) * (Jiz - Jzz) / (np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2) * (
                    (Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2))
        dp0_dJzy = (Jix + Jzx) / ((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2)
        dp1_dJzy = (-Jix + Jzx) / ((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2)

        # Derivatives wrt Jzz
        dw0_dJzz = (Jiz + Jzz) / np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2)
        dw1_dJzz = (-Jiz + Jzz) / np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2)
        dd0_dJzz = np.sqrt((Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2) / (
                    (Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2 + (Jiz + Jzz) ** 2)
        dd1_dJzz = -np.sqrt((Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2) / (
                    (Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2 + (Jiz - Jzz) ** 2)
        dp0_dJzz = 0
        dp1_dJzz = 0

        transform_matrix = np.array([[dw0_dJix, dd0_dJix, dp0_dJix, dw1_dJix, dd1_dJix, dp1_dJix],
                                     [dw0_dJiy, dd0_dJiy, dp0_dJiy, dw1_dJiy, dd1_dJiy, dp1_dJiy],
                                     [dw0_dJiz, dd0_dJiz, dp0_dJiz, dw1_dJiz, dd1_dJiz, dp1_dJiz],
                                     [dw0_dJzx, dd0_dJzx, dp0_dJzx, dw1_dJzx, dd1_dJzx, dp1_dJzx],
                                     [dw0_dJzy, dd0_dJzy, dp0_dJzy, dw1_dJzy, dd1_dJzy, dp1_dJzy],
                                     [dw0_dJzz, dd0_dJzz, dp0_dJzz, dw1_dJzz, dd1_dJzz, dp1_dJzz]])

        return transform_matrix

    def fisher_information(self, n_config, t,
                           FLAG_noise=False,
                           FLAG_normalization=False,
                           type_param='J'):
        '''
        Define function to calculate Fisher Information for different configurations
        param_array1 = (omega0, delta0, phi0, omega1, delta1, phi1)
        param_array2 = (Jix, Jiy, Jiz, Jzx, Jzy, Jzz)

        Learner inputs:
        FLAG_noise - Does the Learner want to consider noise even if present in the SystemModel
        FLAG_normalization - preference of the Learner how the model computations are done
        type_param - J (coefficients of Pauli product terms) or param (Ed's parameterization)

        TODO: Right now this gives Fisher information only for the J parameterization, also include the other
        '''
        pvec = self.model_probabilities(n_config, t,  FLAG_noise=FLAG_noise)
        jacobian = self.jacobian_likelihood(n_config, t, FLAG_noise=FLAG_noise, type_param=type_param)

        if FLAG_normalization:
            if type_param == 'J':
                jacobian = self.xi_J*jacobian
            elif type_param == 'param':
                jacobian = self.xi_param * jacobian
            else:
                raise RuntimeError('Wrong argument for type_param')

        fisher_info_matrix = np.outer(jacobian, jacobian)

        # Hack given the form of the jacobian here -- need to modify for later use
        #denom_fisher_info = 1 / (pvec ** 2 + 1e-10)
        #fisher_info_matrix = np.dot(pvec, denom_fisher_info) * fisher_info_matrix

        # Assuming that the target qubit is measured as usual
        # if FLAG_noise and self.FLAG_readout_noise:
        #     r0, r1 = self.readout_noise
        #     p0 = (1 - r0) * (pvec[0] + pvec[2]) + r1 * (pvec[1] + pvec[3])
        # else:
        #     p0 = pvec[0] + pvec[2]
        p0 = self.target_probability(n_config, t, FLAG_noise=FLAG_noise)

        denom_fisher_info = (1 / (p0 * (1 - p0) + 1e-10))
        fisher_info_matrix = denom_fisher_info*fisher_info_matrix

        return fisher_info_matrix

    def plot_rabi_oscillations(self, n_config, time_stamps, FLAG_noise=False,
                                FLAG_plot=False, FLAG_analytical=False, title_plot='rabi_vs_t.eps'):
        """
        Rabi oscillation for a particular config
        """
        rabi_trend = np.zeros(len(time_stamps))

        for ind_t in range(len(time_stamps)):
            t = time_stamps[ind_t]
            if FLAG_analytical:
                rabi_trend[ind_t] = self.analytical_rabi_oscillations(n_config, t, FLAG_noise=FLAG_noise)
            else:
                rabi_trend[ind_t] = self.rabi_oscillations(n_config, t, FLAG_noise=FLAG_noise)

        # Plot the figure
        if FLAG_plot:
            plt.figure(2, figsize=(10, 6))
            plt.plot(time_stamps, rabi_trend, 'b-')
            #plt.plot(time_stamps, fi_trend, 'bo')
            plt.grid(True)
            plt.xlabel("t")
            plt.ylabel("p(0) - p(1)")
            plt.savefig(title_plot)
            plt.show()

        return rabi_trend

    def plot_fisher_information(self, time_stamps, FLAG_noise=False, FLAG_normalization=False,
                                type_param = 'J',
                                FLAG_plot=True, title_plot='fisher_information2_vs_t.eps'):
        """
        # Look at tr(F) for all the different M and U as a function of time

        TODO: FLAG_Normalization hasn't been included properly
        """

        # tr(F) considering all the configurations at each time instance
        fi_trend = np.zeros(len(time_stamps))

        for ind_t in range(len(time_stamps)):
            t = time_stamps[ind_t]
            fisher_info_temp = np.zeros((6, 6))
            for n_config in range(6):
                fisher_info_temp = fisher_info_temp + self.fisher_information(n_config, t,
                                                                  FLAG_noise=FLAG_noise,
                                                                  FLAG_normalization=FLAG_normalization,
                                                                  type_param=type_param)

            fi_trend[ind_t] = np.trace(fisher_info_temp)

        # Plot the figure
        if FLAG_plot:
            plt.figure(2, figsize=(10, 6))
            plt.plot(time_stamps, fi_trend, 'b-')
            #plt.plot(time_stamps, fi_trend, 'bo')
            plt.grid(True)
            plt.xlabel("t")
            plt.ylabel("tr(F)")
            plt.savefig(title_plot)
            plt.show()

        return fi_trend

    def plot_cramer_rao_bound(self, time_stamps, FLAG_noise=False, FLAG_normalization=False, type_param='J',
                              FLAG_plot=True, title_plot='cramer_rao_bound2_vs_t.eps'):
        """
        Look at tr(F^{-1}) for all the different M and U as a function of time

        TODO: FLAG_Normalization hasn't been included properly
        """

        # CRB considering all the configurations at each time instance
        crb_trend = np.zeros(len(time_stamps))

        for ind_t in range(len(time_stamps)):
            t = time_stamps[ind_t]
            fisher_info_temp = np.zeros((6, 6))
            for n_config in range(6):
                fisher_info_temp = fisher_info_temp + self.fisher_information(n_config, t,
                                                                  FLAG_noise=FLAG_noise,
                                                                  FLAG_normalization=FLAG_normalization,
                                                                  type_param=type_param)

            # import pdb; pdb.set_trace()

            if np.linalg.matrix_rank(fisher_info_temp) < 3:
                print('Singular Fisher Information Matrix')
                crb_trend[ind_t] = 1e4
            else:
                print('Non-Singular Fisher Information Matrix')
                crb_trend[ind_t] = np.trace(np.linalg.inv(fisher_info_temp))

        # Plot the figure
        if FLAG_plot:
            plt.figure(2, figsize=(10, 6))
            plt.plot(time_stamps, crb_trend, 'b-')
            plt.plot(time_stamps, crb_trend, 'bo')
            plt.grid(True)
            plt.xlabel("t")
            plt.ylabel(r"tr($F^{-1})$)")
            plt.savefig(title_plot)
            plt.show()

        return crb_trend

    # Look at tr(F^{-1}) for all the different M and U as a function of time
    def plot_cramer_rao_bound_bins(self, time_stamps, FLAG_noise=False, FLAG_normalization=False, type_param='J',
                                   len_bin=3, FLAG_plot=True,
                                   title_plot='crb2_bins_vs_t.eps'):

        # len_bin corresponds to the length covered by each bin
        n_bins = int(np.ceil(len(time_stamps) / len_bin))

        # CRB considering all the configurations at each time instance
        crb_trend = np.zeros(n_bins)

        for ind_bin in range(n_bins):
            fisher_info_temp = np.zeros((6, 6))
            time_range = time_stamps[ind_bin * len_bin:(ind_bin + 1) * len_bin]
            for ind_t in range(len(time_range)):
                t = time_range[ind_t]

                for n_config in range(6):
                    fisher_info_temp = fisher_info_temp + self.fisher_information(n_config, t,
                                                                                  FLAG_noise=FLAG_noise,
                                                                                  FLAG_normalization=FLAG_normalization,
                                                                                  type_param=type_param)

            if np.linalg.matrix_rank(fisher_info_temp) < 3:
                print('Singular Fisher Information Matrix')
                crb_trend[ind_bin] = np.nan
            else:
                print('Non-Singular Fisher Information Matrix')
                crb_trend[ind_bin] = np.trace(np.linalg.inv(fisher_info_temp))

        # Plotting
        xticks_index = np.arange(n_bins)
        xticks_labels = []
        for ind_bin in range(n_bins):
            temp_label = str(ind_bin * len_bin) + '-' + str((ind_bin + 1) * len_bin - 1)
            xticks_labels.append(temp_label)

        if FLAG_plot:
            plt.figure(2, figsize=(10, 6))
            plt.plot(np.arange(n_bins), crb_trend, 'b-')
            plt.plot(np.arange(n_bins), crb_trend, 'bo')
            plt.grid(True)
            plt.xlabel("Bins of time stamps")
            plt.ylabel(r"tr($F^{-1}$)")
            plt.xticks(xticks_index, xticks_labels, fontsize=8, rotation=30)
            plt.savefig(title_plot)
            plt.show()

        return crb_trend

    # Define function to calculate Shannon Entropy
    def shannon_entropy(self, n_config, t, FLAG_noise=False):
        pvec = self.model_probabilities(n_config, t, FLAG_noise=FLAG_noise)
        entropy = -np.dot(pvec, np.log(pvec + 1e-10))

        return entropy

    # plot of Shannon entropy
    def plot_shannon_entropy(self, time_stamps, FLAG_noise=False, FLAG_plot=True):
        # Entropy considering all the configurations at each time instance
        entropy_trend = [np.zeros(len(time_stamps)) for ind_config in range(6)]

        for ind_config in range(6):
            entropy_trend[ind_config] = [self.shannon_entropy(ind_config, t, FLAG_noise=FLAG_noise) for t in time_stamps]

        # Plot the trends
        color_lines = ['r-', 'b-', 'g-', 'c-', 'y-', 'm-']
        markers_lines = ['ro', 'bs', 'gd', 'cv', 'y^', 'm+']
        labels_lines = [str(n) for n in range(6)]

        if FLAG_plot:
            plt.figure(2, figsize=(16, 10))

            plt.subplot(211)
            for ind_config in [0, 2, 4]:
                plt.plot(time_stamps, entropy_trend[ind_config], color_lines[ind_config],
                         label=labels_lines[ind_config])
                plt.plot(time_stamps, entropy_trend[ind_config], markers_lines[ind_config])
            plt.grid(True)
            #xlabel_fig = 't [s]'
            #plt.xlabel(xlabel_fig)
            plt.ylabel("Shannon Entropy")
            plt.title(r'Control in $|0 \rangle$')
            plt.legend(loc='upper right')

            plt.subplot(212)
            for ind_config in [1, 3, 5]:
                plt.plot(time_stamps, entropy_trend[ind_config], color_lines[ind_config],
                         label=labels_lines[ind_config])
                plt.plot(time_stamps, entropy_trend[ind_config], markers_lines[ind_config])
            plt.grid(True)
            xlabel_fig = 't [s]'
            plt.xlabel(xlabel_fig)
            plt.ylabel("Shannon Entropy")
            plt.title(r'Control in $|1 \rangle$')
            plt.legend(loc='upper right')

            plt.show()

        return entropy_trend

    def filter_actions_entropy(self, list_actions, time_stamps,
                               xi_t=1e-7, FLAG_noise=False, threshold=0.5, FLAG_filter=True):
        """
        Don't return a subset of actions but rather send a query distribution with 1 where the action can be taken
        and zero otherwise. Basically does the same thing we desire overall. And I think then we can use
        Fisher information query as it is to compute the fisher information

        Inputs:
            FLAG_filter (bool): Should we filter or release a query distribution
        """
        entropy_trend = self.plot_shannon_entropy(time_stamps, FLAG_noise=FLAG_noise, FLAG_plot=False)
        entropy_trend = np.array(entropy_trend)
        n_actions = len(list_actions)

        entropy_threshold = threshold*np.amax(entropy_trend, axis=1) + (1-threshold)*np.amin(entropy_trend, axis=1)

        q_entropy = np.zeros(n_actions)
        for ind_action in range(n_actions):
            action_temp = list_actions[ind_action]
            n_config_temp = 2*action_temp[0] + action_temp[1]

            if self.shannon_entropy(n_config_temp, action_temp[2]*xi_t, FLAG_noise=FLAG_noise) >= entropy_threshold[n_config_temp]:
                q_entropy[ind_action] = 1

        # Indices of actions that are high in entropy
        ind_pruned_actions = np.where(np.isclose(q_entropy, 1))[0]

        if FLAG_filter:
            # Prune actions and return
            pruned_actions = [list_actions[ind] for ind in ind_pruned_actions]

            return pruned_actions, ind_pruned_actions
        else:
            # Return a "query distribution" of sorts
            return q_entropy, ind_pruned_actions

    def filter_actions_rb_crossings(self, list_actions, time_stamps, xi_t=1e-7, FLAG_noise=False, threshold=5e-2):
        """
        Don't return a subset of actions but rather send a query distribution with 1 where the action can be taken
        and zero otherwise. Basically does the same thing we desire overall. And I think then we can use
        Fisher information query as it is to compute the fisher information
        """
        # Get the rabi oscillations for the given actions
        N_configs = 6
        rabi_trend = np.zeros(shape=(N_configs, len(time_stamps)))

        for ind_config in range(N_configs):
            rabi_trend[ind_config,:] = self.plot_rabi_oscillations(ind_config, time_stamps, FLAG_noise=FLAG_noise,
                                                                   FLAG_plot=False)

        n_actions = len(list_actions)

        # Only keep those actions where the crossings are close to zero
        q_rabi = np.zeros(n_actions)
        for ind_action in range(n_actions):
            action_temp = list_actions[ind_action]
            n_config_temp = 2*action_temp[0] + action_temp[1]

            if np.abs(self.rabi_oscillations(n_config_temp, action_temp[2]*xi_t, FLAG_noise=FLAG_noise)) <= threshold:
                q_rabi[ind_action] = 1

        return q_rabi

    # plot of Shannon entropy
    def plot_total_shannon_entropy(self, time_stamps, FLAG_noise=False, FLAG_plot=True):
        # Entropy considering all the configurations at each time instance
        entropy_trend = np.zeros(len(time_stamps))

        for ind_t in range(len(time_stamps)):
            t = time_stamps[ind_t]
            for n_config in range(6):
                entropy_trend[ind_t] += self.shannon_entropy(n_config, t, FLAG_noise=FLAG_noise)

        # Plot the figure
        if FLAG_plot:
            plt.figure(2, figsize=(16, 6))
            plt.plot(time_stamps, entropy_trend, 'b-')
            plt.plot(time_stamps, entropy_trend, 'bo')
            plt.grid(True)
            xlabel_fig = 't [s]'
            plt.xlabel(xlabel_fig)
            plt.ylabel("Shannon Entropy")
            #plt.savefig(title_plot)
            plt.show()

        return entropy_trend
