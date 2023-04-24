import numpy as np
import math
import cmath
from scipy.optimize import minimize
import scipy.fftpack
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar, curve_fit, leastsq
import os

# Running tensorflow in compat mode
import tensorflow
if int(tensorflow.__version__[0]) > 1:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf

# Set number of threads of tensorflow
tf_num_threads=1
tf.config.threading.set_inter_op_parallelism_threads(tf_num_threads)

# other optimization imports
from . import fft_estimators, linear_estimators

# plotting
import matplotlib.pyplot as plt


class InitialEstimator(object):
    """
    Sets up the Initial Estimator with different solver options and loss functions

    InitialEstimator uses the rabi oscillations for estimation

    TODO: Add options for different parameterizations and loss functions
    """

    def __init__(self, syn_data, env_cr=None, A_cr=None,
                 FLAG_mle_solver=False, solver_options=None, do_plot=False,
                 FLAG_fft_window=False, type_fft_window=None, FLAG_fft_high_resoultion=False):

        # Regarding the use of FFT
        self.FLAG_fft_window = FLAG_fft_window
        self.type_fft_window = type_fft_window
        self.FLAG_fft_high_resolution = FLAG_fft_high_resoultion

        # Regarding the use of data
        samples = np.array(syn_data['samples']).astype(np.float32)
        mvec = np.array(syn_data['mvec']).astype(np.int16)
        uvec = np.array(syn_data['uvec']).astype(np.int16)
        tvec = np.array(syn_data['tvec']).astype(np.float32)

        # This is really dirty but I think this is the easiest way to get this working for the moment
        self.samples_config = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        self.mvec_config = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        self.uvec_config = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        self.tvec_config = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

        for ind_sample in range(len(mvec)):
            ind_config = int(2*mvec[ind_sample] + uvec[ind_sample])

            self.samples_config[ind_config].extend([samples[ind_sample]])
            self.mvec_config[ind_config].extend([mvec[ind_sample]])
            self.uvec_config[ind_config].extend([uvec[ind_sample]])
            self.tvec_config[ind_config].extend([tvec[ind_sample]])

        for ind_config in range(6):
            self.samples_config[ind_config] = np.array(self.samples_config[ind_config]).astype(np.float32)
            self.mvec_config[ind_config] = np.array(self.mvec_config[ind_config]).astype(np.int16)
            self.uvec_config[ind_config] = np.array(self.uvec_config[ind_config]).astype(np.int16)
            self.tvec_config[ind_config] = np.array(self.tvec_config[ind_config]).astype(np.float32)

        self.FLAG_classification = syn_data['FLAG_classification']

        # Extract relevant parameters
        time_stamps = A_cr.tset * A_cr.xi_t
        freq_convert = A_cr.freq_convert

        # Calculate the rabi oscillations from the given data
        pvec_data = A_cr.rabi_data(env_cr)
        nshots_rabi_data, n0shots_rabi_data = A_cr.nshots_rabi_data(env_cr)

        # Get the sample frequency
        if 'sample_freq' in syn_data.keys():
            self.sample_freq = syn_data['sample_freq']
        else:
            first_time_stamp = np.amin(time_stamps)
            last_time_stamp = np.amax(time_stamps)

            delta_time = (last_time_stamp - first_time_stamp) / (len(time_stamps) - 1)
            self.sample_freq = 1.0 / delta_time

        self.time_stamps = time_stamps
        self.pvec_data = pvec_data
        self.nshots_rabi_data = nshots_rabi_data
        self.n0shots_rabi_data = n0shots_rabi_data
        self.freq_convert = freq_convert
        self.do_plot = do_plot

        if FLAG_mle_solver:
            # Setup solver options
            default_solver_options = {'nepochs': 200, 'neval_period': 10,
                                      'learning_rate': 0.001, 'optimizer': 'adam',
                                      'mini_batch_size': 100}

            if solver_options is None:
                self.mle_nepochs = default_solver_options['nepochs']
                self.mle_neval_period = default_solver_options['neval_period']
                self.mle_learning_rate = default_solver_options['learning_rate']
                self.mle_optimizer = default_solver_options['optimizer']
                self.mle_mini_batch_size = default_solver_options['mini_batch_size']
            else:
                if 'nepochs' in solver_options.keys():
                    self.mle_nepochs = solver_options['nepochs']
                else:
                    self.mle_nepochs = default_solver_options['nepochs']

                if 'neval_period' in solver_options.keys():
                    self.mle_neval_period = solver_options['neval_period']
                else:
                    self.mle_neval_period = default_solver_options['neval_period']

                if 'learning_rate' in solver_options.keys():
                    self.mle_learning_rate = solver_options['learning_rate']
                else:
                    self.mle_learning_rate = default_solver_options['learning_rate']

                if 'optimizer' in solver_options.keys():
                    self.mle_optimizer = solver_options['optimizer']
                else:
                    self.mle_optimizer = default_solver_options['optimizer']

                if 'mini_batch_size' in solver_options.keys():
                    self.mle_mini_batch_size = solver_options['mini_batch_size']
                else:
                    self.mle_mini_batch_size = default_solver_options['mini_batch_size']

    def rabi_amplitude_mle_estimator(self, init_param, rabi_freq, ind_config, verbose=True, do_plot=False):
        '''
        MLE Estimator for rabi amplitudes A, B, C given value of rabi frequency
        '''
        if verbose:
            print("Initial Estimation: Model training for %s epochs, with evaluation every %s steps" % (
            self.mle_nepochs, self.mle_neval_period))

        # Extract solver options
        nepochs = self.mle_nepochs
        neval_period = self.mle_neval_period
        learning_rate = self.mle_learning_rate
        mini_batch_size = self.mle_mini_batch_size

        xy = np.stack([self.tvec_config[ind_config], self.samples_config[ind_config]], axis=1)
        batch_size = len(self.tvec_config[ind_config])  # MLE Batch Size

        n_mini_batches = int(xy.shape[0] / mini_batch_size)

        # graph input
        T = tf.placeholder(tf.float32, name="X_time_points")
        Y = tf.placeholder(tf.float32, name="Y_samples")  # y = 0 or 1
        XI = tf.placeholder(tf.float32, name="XI_param")  # y = 0 or 1
        RABI_FREQ = tf.placeholder(tf.float32, name="rabi_freq")  # y = 0 or 1

        # model variables
        theta_nd = tf.Variable(init_param, name="theta_nd", dtype=tf.float32)

        if verbose:
            print("Input data xy shape=%s" % str(xy.shape))
            print("Each epoch has %d mini_batches of size %s" % (n_mini_batches, mini_batch_size))
            print("Input data shapes samples=%s, tpts=%s" % (self.samples_config[ind_config].shape, self.tvec_config[ind_config].shape))
            print("Input data placeholders X=%s, Y=%s, XI=%s" % (T, Y, XI))
            print("initial values: theta=%s" % (init_param,))

        # model to predict sample probability (prob(k=0))
        Y_sp = tf_model_free_probabilities(theta_nd, RABI_FREQ, T)

        # loss function (binary cross-entropy)
        loss = -1 * tf.log((1 - Y) * Y_sp + Y * (1 - Y_sp) + 1.0e-10)  # log likelihood
        loss = tf.reduce_mean(loss)  # take mean over batch

        # optimizer
        if self.mle_optimizer == "gd":
            gdo = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optimizer_op = gdo.minimize(loss)
            op_name = "GD"
        elif self.mle_optimizer == "adagrad":
            optimizer_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
            op_name = "Adagrad"
        else:
            optimizer_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            op_name = "Adam"

        losses = []  # record losses at each epoch step
        steps = []

        # run MLE
        if verbose:
            print("Using %s optimmizer, learning rate=%s" % (op_name, learning_rate))
            print("Running MLE over %s datapoints with %s epochs" % (batch_size, nepochs))

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for k in range(nepochs):
                np.random.shuffle(xy)  # random in-place permutation of first dimension
                for n in range(n_mini_batches):
                    n0 = n * mini_batch_size
                    sess.run(optimizer_op, feed_dict={T: xy[n0:n0 + mini_batch_size, 0],
                                                      Y: xy[n0:n0 + mini_batch_size, 1],
                                                      RABI_FREQ: rabi_freq})

                if not (k % neval_period):
                    results = sess.run([loss, theta_nd],
                                       feed_dict={T: self.tvec_config[ind_config],
                                                  Y: self.samples_config[ind_config],
                                                  RABI_FREQ: rabi_freq})
                    if verbose:
                        print("    Epoch %s: loss=%s, theta=%s" % tuple([k] + results))
                    losses.append(results[0])
                    steps.append(k)
                    if np.isnan(results[0]):
                        raise Exception("loss is NaN, quitting!")

            results = sess.run([loss, theta_nd],
                               feed_dict={T: self.tvec_config[ind_config],
                                          Y: self.samples_config[ind_config],
                                          RABI_FREQ: rabi_freq})

        m_loss, m_theta = results

        if do_plot:
            plt.figure(2, figsize=(10, 6))
            plt.plot(steps, losses, 'go')
            plt.plot(steps, losses)
            plt.grid(True)
            plt.xlabel("Epoch step number")
            plt.ylabel("Loss (negative log likelihood)")
            plt.suptitle("Tensorflow MLE on dataset with %s samples using %s optimizer" %
                         (samples.size, op_name))

        ds = {'loss': losses, 'steps': steps, 'results': results}

        return ds

    def normal_equations(self, omega, f):
        R00 = np.sum([ np.cos(omega*t)**2 for t in self.time_stamps ])
        R01 = np.sum([ np.cos(omega*t)*np.sin(omega*t) for t in self.time_stamps ])
        R02 = np.sum([np.cos(omega * t) for t in self.time_stamps])
        R11 = np.sum([np.sin(omega * t) ** 2 for t in self.time_stamps])
        R12 = np.sum([np.sin(omega * t) for t in self.time_stamps])
        R22 = np.sum([1 for t in self.time_stamps])

        R = np.array([[R00, R01, R02], [R01, R11, R12], [R02, R12, R22]])

        x = np.linalg.solve(R, f)

        # Ref: https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
        return np.vdot(x,f)

    def normal_equations_uncertainty(self, omega, f, var_x):
        cos_omega_t = np.array([np.cos(omega*t) for t in self.time_stamps])
        sin_omega_t = np.array([np.sin(omega*t) for t in self.time_stamps])
        temp_t = np.array(1/var_x)

        R00 = np.dot(cos_omega_t**2,temp_t)
        R01 = np.dot(cos_omega_t * sin_omega_t, temp_t)
        R02 = np.dot(cos_omega_t, temp_t)
        R11 = np.dot(sin_omega_t**2, temp_t)
        R12 = np.dot(sin_omega_t, temp_t)
        R22 = np.sum(temp_t)

        R = np.array([[R00, R01, R02], [R01, R11, R12], [R02, R12, R22]])

        x = np.linalg.solve(R, f)

        # Ref: https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
        return np.vdot(x,f)

    def plot_spectrum(self, omega_array, power_spectrum, rabi_chisq, omega_truth=None):
        plt.figure(2, figsize=(10, 7))
        freq_plot = np.copy(omega_array)
        rabi_power_plot = np.copy(power_spectrum)
        rabi_chisq_plot = np.copy(rabi_chisq)
        xlabel_plot = 'Frequency'

        plt.subplot(311)
        plt.title(r'Blue = Control in $|0\rangle$,  Red = Control in $|1\rangle$')
        plt.plot(freq_plot, rabi_power_plot[0, :], 'bo-')
        plt.plot(freq_plot, rabi_power_plot[1, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        # plt.yscale('log')
        plt.ylabel(r'$\langle X \rangle$')

        plt.subplot(312)
        plt.plot(freq_plot, rabi_power_plot[2, :], 'bo-')
        plt.plot(freq_plot, rabi_power_plot[3, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        # plt.yscale('log')
        plt.ylabel(r'$\langle Y \rangle$')

        plt.subplot(313)
        plt.plot(freq_plot, rabi_power_plot[4, :], 'bo-')
        plt.plot(freq_plot, rabi_power_plot[5, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        # plt.yscale('log')
        plt.ylabel(r'$\langle Z \rangle$')

        plt.xlabel(xlabel_plot)

        plt.show()

        # Plot chisq
        plt.figure(3, figsize=(10, 3))
        # plt.title('Blue = Control in |0> max @ '
        #          + str(np.argmax(rabi_chisq[0,:int((1 + rabi_chisq.shape[0]) / 2)]))
        #          + ',  Red = Control in |1> max @ '
        #          + str(np.argmax(rabi_chisq[1,:int((1 + rabi_chisq.shape[1]) / 2)])))

        plt.plot(freq_plot, rabi_chisq_plot[0, :], 'bo-')
        plt.plot(freq_plot, rabi_chisq_plot[1, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        plt.ylabel('Sum Magnitude Squared')
        plt.xlabel(xlabel_plot)

        plt.show()

    def corrected_power_spectrum(self, rabi_data=None, prior_freq=[0,1e8], do_plot=False, omega_truth=None):
        '''
        Solve the normal equations as laid out in Ed's slides

        TODO: Implement for the rotating exponential
        '''
        if rabi_data is None:
            rabi_data = self.pvec_data

        # Get the fourier transform of the rabi_data
        rabi_fourier, _, _ = fft_estimators.fft_rabi_data(rabi_data, 6)

        freq_temp = self.freq_convert*np.arange(rabi_data.shape[1])
        ind_cut = np.where(freq_temp > prior_freq[1])[0][0]

        rabi_f0 = rabi_fourier[:,0]

        if prior_freq is not None:
            ind_cut = np.where(freq_temp > prior_freq[1])[0][0]
            rabi_fourier = rabi_fourier[:,1:ind_cut]
            omega_array = freq_temp[1:ind_cut]
        else:
            omega_array = freq_temp[1:ind_cut]

        power_spectrum = np.ndarray(shape=rabi_fourier.shape, dtype=float)

        counter_omega = 0
        for omega in omega_array:
            for ind_config in range(6):
                f_temp = np.array([np.real(rabi_fourier[ind_config, counter_omega]),
                                   -np.imag(rabi_fourier[ind_config, counter_omega]),
                                   rabi_f0[ind_config]])

                power_spectrum[ind_config, counter_omega] = self.normal_equations(omega, f_temp)

            counter_omega += 1

        # Combine power series to form chi-squared specra for frequency identification
        rabi_chisq = np.zeros(shape=(2, rabi_fourier.shape[1]))
        rabi_chisq[0, :] = power_spectrum[0, :] + power_spectrum[2, :] + power_spectrum[4, :]
        rabi_chisq[1, :] = power_spectrum[1, :] + power_spectrum[3, :] + power_spectrum[5, :]

        if do_plot:
            self.plot_spectrum(omega_array, power_spectrum, rabi_chisq, omega_truth=omega_truth)

        return power_spectrum, rabi_chisq, omega_array

    def corrected_power_spectrum_uncertainty(self, rabi_data=None, prior_freq=[0,1e8], do_plot=False, omega_truth=None):
        '''
        Solve the normal equations as laid out in Ed's slides

        TODO: Implement for the rotating exponential
        '''
        if rabi_data is None:
            rabi_data = self.pvec_data

        # Get the fourier transform of the rabi_data
        alpha_rabi_data = self.n0shots_rabi_data + 1 / 2
        beta_rabi_data = (self.nshots_rabi_data - self.n0shots_rabi_data) + 1 / 2

        var_x = alpha_rabi_data*beta_rabi_data/ ( ((self.nshots_rabi_data + 1)**2) * (self.nshots_rabi_data + 2) )

        rabi_fourier, _, _ = fft_estimators.fft_rabi_data(rabi_data/var_x, 6)

        freq_temp = self.freq_convert*np.arange(rabi_data.shape[1])
        ind_cut = np.where(freq_temp > prior_freq[1])[0][0]

        rabi_f0 = rabi_fourier[:,0]

        if prior_freq is not None:
            ind_cut = np.where(freq_temp > prior_freq[1])[0][0]
            rabi_fourier = rabi_fourier[:,1:ind_cut]
            omega_array = freq_temp[1:ind_cut]
        else:
            omega_array = freq_temp[1:ind_cut]

        power_spectrum = np.ndarray(shape=rabi_fourier.shape, dtype=float)

        counter_omega = 0
        for omega in omega_array:
            for ind_config in range(6):
                f_temp = np.array([np.real(rabi_fourier[ind_config, counter_omega]),
                                   -np.imag(rabi_fourier[ind_config, counter_omega]),
                                   rabi_f0[ind_config]])

                power_spectrum[ind_config, counter_omega] = self.normal_equations_uncertainty(omega, f_temp, var_x[ind_config,:])

            counter_omega += 1

        # Combine power series to form chi-squared specra for frequency identification
        rabi_chisq = np.zeros(shape=(2, rabi_fourier.shape[1]))
        rabi_chisq[0, :] = power_spectrum[0, :] + power_spectrum[2, :] + power_spectrum[4, :]
        rabi_chisq[1, :] = power_spectrum[1, :] + power_spectrum[3, :] + power_spectrum[5, :]

        if do_plot:
            self.plot_spectrum(omega_array, power_spectrum, rabi_chisq, omega_truth=omega_truth)

        return power_spectrum, rabi_chisq, omega_array

    def corrected_power_spectrum_halfway(self, rabi_data=None, prior_freq=[0, 1e8], do_plot=False, omega_truth=None):
        '''
        Solve the normal equations as laid out in Ed's slides

        TODO: Implement for the rotating exponential
        '''
        if rabi_data is None:
            rabi_data = self.pvec_data

        n_configs, n_t = rabi_data.shape

        #rotating_exponential = np.array([np.exp(-1j * 2 * np.pi * self.sample_freq / (2 * n_t) * k) for k in range(n_t)])
        rotating_exponential = np.array([np.exp(-1j * 2 * np.pi * 1 / (2 * n_t) * k) for k in range(n_t)])
        #rotating_exponential = np.array([np.exp(-1j * 2 * np.pi * sample_freq / (2 * n_t) * _t) for _t in tvec])

        rabi_data_rot = np.zeros(shape=rabi_data.shape, dtype=complex)
        for ind_config in range(n_configs):
            rabi_data_rot[ind_config, :] = rabi_data[ind_config, :] * rotating_exponential

        # Get the fourier transform of the rabi_data
        rabi_fourier, _, _ = fft_rabi_data(rabi_data_rot, 6)

        freq_temp = self.freq_convert*(np.arange(rabi_data.shape[1]) + 1/2)
        ind_cut = np.where(freq_temp > prior_freq[1])[0][0]

        rabi_f0 = rabi_fourier[:,0]

        if prior_freq is not None:
            ind_cut = np.where(freq_temp > prior_freq[1])[0][0]
            rabi_fourier = rabi_fourier[:,0:ind_cut]
            omega_array = freq_temp[0:ind_cut]
        else:
            omega_array = freq_temp

        power_spectrum = np.ndarray(shape=rabi_fourier.shape, dtype=float)

        counter_omega = 0
        for omega in omega_array:
            for ind_config in range(6):
                f_temp = np.array([np.real(rabi_fourier[ind_config, counter_omega]),
                                   -np.imag(rabi_fourier[ind_config, counter_omega]),
                                   rabi_f0[ind_config]])

                power_spectrum[ind_config, counter_omega] = self.normal_equations(omega, f_temp)

            counter_omega += 1

        # Combine power series to form chi-squared specra for frequency identification
        rabi_chisq = np.zeros(shape=(2, rabi_fourier.shape[1]))
        rabi_chisq[0, :] = power_spectrum[0, :] + power_spectrum[2, :] + power_spectrum[4, :]
        rabi_chisq[1, :] = power_spectrum[1, :] + power_spectrum[3, :] + power_spectrum[5, :]

        if do_plot:
            self.plot_spectrum(omega_array, power_spectrum, rabi_chisq)

        return power_spectrum, rabi_chisq, omega_array

    def corrected_power_spectrum_halfway_uncertainty(self, rabi_data=None, prior_freq=[0,1e8], do_plot=False, omega_truth=None):
        '''
        Solve the normal equations as laid out in Ed's slides

        TODO: Implement for the rotating exponential
        '''
        if rabi_data is None:
            rabi_data = self.pvec_data

        # Get the fourier transform of the rabi_data
        alpha_rabi_data = self.n0shots_rabi_data + 1 / 2
        beta_rabi_data = (self.nshots_rabi_data - self.n0shots_rabi_data) + 1 / 2

        var_x = alpha_rabi_data*beta_rabi_data/ ( ((self.nshots_rabi_data + 1)**2) * (self.nshots_rabi_data + 2) )

        n_configs, n_t = rabi_data.shape

        # rotating_exponential = np.array([np.exp(-1j * 2 * np.pi * self.sample_freq / (2 * n_t) * k) for k in range(n_t)])
        rotating_exponential = np.array([np.exp(-1j * 2 * np.pi * 1 / (2 * n_t) * k) for k in range(n_t)])
        # rotating_exponential = np.array([np.exp(-1j * 2 * np.pi * sample_freq / (2 * n_t) * _t) for _t in tvec])

        rabi_data_rot = np.zeros(shape=rabi_data.shape, dtype=complex)
        for ind_config in range(n_configs):
            rabi_data_rot[ind_config, :] = rabi_data[ind_config, :] * rotating_exponential

        # Get the fourier transform of the rabi_data
        rabi_fourier, _, _ = fft_rabi_data(rabi_data_rot/var_x, 6)

        freq_temp = self.freq_convert * (np.arange(rabi_data.shape[1]) + 1 / 2)
        ind_cut = np.where(freq_temp > prior_freq[1])[0][0]

        rabi_f0 = rabi_fourier[:, 0]

        if prior_freq is not None:
            ind_cut = np.where(freq_temp > prior_freq[1])[0][0]
            rabi_fourier = rabi_fourier[:, 0:ind_cut]
            omega_array = freq_temp[0:ind_cut]
        else:
            omega_array = freq_temp

        power_spectrum = np.ndarray(shape=rabi_fourier.shape, dtype=float)

        counter_omega = 0
        for omega in omega_array:
            for ind_config in range(6):
                f_temp = np.array([np.real(rabi_fourier[ind_config, counter_omega]),
                                   -np.imag(rabi_fourier[ind_config, counter_omega]),
                                   rabi_f0[ind_config]])

                power_spectrum[ind_config, counter_omega] = self.normal_equations_uncertainty(omega, f_temp, var_x[ind_config,:])

            counter_omega += 1

        # Combine power series to form chi-squared specra for frequency identification
        rabi_chisq = np.zeros(shape=(2, rabi_fourier.shape[1]))
        rabi_chisq[0, :] = power_spectrum[0, :] + power_spectrum[2, :] + power_spectrum[4, :]
        rabi_chisq[1, :] = power_spectrum[1, :] + power_spectrum[3, :] + power_spectrum[5, :]

        if do_plot:
            self.plot_spectrum(omega_array, power_spectrum, rabi_chisq, omega_truth=omega_truth)

        return power_spectrum, rabi_chisq, omega_array

    def plot_corrected_power_spectrum(self, prior_freq=[0,5e7], omega_truth=None):
        # Normal FFT + Higher Resolution FFT
        rabi_power0, rabi_chisq0, omega_array0 = self.corrected_power_spectrum(prior_freq=prior_freq)
        rabi_power1, rabi_chisq1, omega_array1 = self.corrected_power_spectrum_halfway(prior_freq=prior_freq)

        # Sorting the collected arrays
        omega_array = np.concatenate((omega_array0, omega_array1))
        rabi_power = np.concatenate((rabi_power0, rabi_power1), axis=1)
        rabi_chisq = np.concatenate((rabi_chisq0, rabi_chisq1), axis=1)

        omega_sorted = np.sort(omega_array)
        sort_omega_index = np.argsort(omega_array)
        rabi_power_sorted = rabi_power[:, sort_omega_index]
        rabi_chisq_sorted = rabi_chisq[:, sort_omega_index]

        # Plot the Power Spectrum
        plt.figure(2, figsize=(10, 7))
        freq_plot = np.copy(omega_sorted)
        rabi_power_plot = np.copy(rabi_power_sorted)
        rabi_chisq_plot = np.copy(rabi_chisq_sorted)
        xlabel_plot = 'Frequency'

        if freq_plot[0] == 0:
            ind_start = 1
        else:
            ind_start = 0

        rabi_freq = np.array([
            freq_plot[np.argmax(rabi_chisq_plot[0, ind_start:int((rabi_chisq_plot.shape[1] - 1) / 2)]) + ind_start],
            freq_plot[np.argmax(rabi_chisq_plot[1, ind_start:int((rabi_chisq_plot.shape[1] - 1) / 2)]) + ind_start]])

        plt.subplot(311)
        plt.title(r'Blue = Control in $|0\rangle$,  Red = Control in $|1\rangle$')
        plt.semilogx(freq_plot, rabi_power_plot[0, :], 'bo-')
        plt.semilogx(freq_plot, rabi_power_plot[1, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        # plt.yscale('log')
        plt.ylabel(r'$\langle X \rangle$')

        plt.subplot(312)
        plt.semilogx(freq_plot, rabi_power_plot[2, :], 'bo-')
        plt.semilogx(freq_plot, rabi_power_plot[3, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        # plt.yscale('log')
        plt.ylabel(r'$\langle Y \rangle$')

        plt.subplot(313)
        plt.semilogx(freq_plot, rabi_power_plot[4, :], 'bo-')
        plt.semilogx(freq_plot, rabi_power_plot[5, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        # plt.yscale('log')
        plt.ylabel(r'$\langle Z \rangle$')

        plt.xlabel(xlabel_plot)

        plt.show()

        # Plot chisq
        plt.figure(3, figsize=(10, 3))
        # plt.title('Blue = Control in |0> max @ '
        #          + str(np.argmax(rabi_chisq[0,:int((1 + rabi_chisq.shape[0]) / 2)]))
        #          + ',  Red = Control in |1> max @ '
        #          + str(np.argmax(rabi_chisq[1,:int((1 + rabi_chisq.shape[1]) / 2)])))

        plt.semilogx(freq_plot, rabi_chisq_plot[0, :], 'bo-')
        plt.semilogx(freq_plot, rabi_chisq_plot[1, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        plt.ylabel('Sum Magnitude Squared')
        plt.xlabel(xlabel_plot)

        plt.show()

        print(rabi_freq)
        if omega_truth is not None:
            print([np.abs(rabi_freq[0] - omega_truth[0]) / omega_truth[0],
                   np.abs(rabi_freq[1] - omega_truth[1]) / omega_truth[1]])

    def plot_corrected_power_spectrum_uncertainty(self, prior_freq=[0,1e8], omega_truth=None):
        # Normal FFT + Higher Resolution FFT
        rabi_power0, rabi_chisq0, omega_array0 = self.corrected_power_spectrum_uncertainty()
        rabi_power1, rabi_chisq1, omega_array1 = self.corrected_power_spectrum_halfway_uncertainty()

        # Sorting the collected arrays
        omega_array = np.concatenate((omega_array0, omega_array1))
        rabi_power = np.concatenate((rabi_power0, rabi_power1), axis=1)
        rabi_chisq = np.concatenate((rabi_chisq0, rabi_chisq1), axis=1)

        omega_sorted = np.sort(omega_array)
        sort_omega_index = np.argsort(omega_array)
        rabi_power_sorted = rabi_power[:, sort_omega_index]
        rabi_chisq_sorted = rabi_chisq[:, sort_omega_index]

        # Plot the Power Spectrum
        plt.figure(2, figsize=(10, 7))
        freq_plot = np.copy(omega_sorted)
        rabi_power_plot = np.copy(rabi_power_sorted)
        rabi_chisq_plot = np.copy(rabi_chisq_sorted)
        xlabel_plot = 'Frequency'

        plt.subplot(311)
        plt.title(r'Blue = Control in $|0\rangle$,  Red = Control in $|1\rangle$')
        plt.plot(freq_plot, rabi_power_plot[0, :], 'bo-')
        plt.plot(freq_plot, rabi_power_plot[1, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        # plt.yscale('log')
        plt.ylabel(r'$\langle X \rangle$')

        plt.subplot(312)
        plt.plot(freq_plot, rabi_power_plot[2, :], 'bo-')
        plt.plot(freq_plot, rabi_power_plot[3, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        # plt.yscale('log')
        plt.ylabel(r'$\langle Y \rangle$')

        plt.subplot(313)
        plt.plot(freq_plot, rabi_power_plot[4, :], 'bo-')
        plt.plot(freq_plot, rabi_power_plot[5, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        # plt.yscale('log')
        plt.ylabel(r'$\langle Z \rangle$')

        plt.xlabel(xlabel_plot)

        plt.show()

        # Plot chisq
        plt.figure(3, figsize=(10, 3))
        # plt.title('Blue = Control in |0> max @ '
        #          + str(np.argmax(rabi_chisq[0,:int((1 + rabi_chisq.shape[0]) / 2)]))
        #          + ',  Red = Control in |1> max @ '
        #          + str(np.argmax(rabi_chisq[1,:int((1 + rabi_chisq.shape[1]) / 2)])))

        plt.plot(freq_plot, rabi_chisq_plot[0, :], 'bo-')
        plt.plot(freq_plot, rabi_chisq_plot[1, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        plt.ylabel('Sum Magnitude Squared')
        plt.xlabel(xlabel_plot)

        plt.show()

    def power_spectrum_linear_regression(self, omega_array, rabi_data=None, prior_freq=None):
        '''
        Assuming omega_array is sorted
        '''
        if rabi_data is None:
            rabi_data = self.pvec_data

        if prior_freq is not None:
            ind_cut = np.where(omega_array > prior_freq[1])[0][0]
            omega_vec = omega_array[0:ind_cut]
        else:
            omega_vec = np.copy(omega_array)

        # Get the power computed from the linear regression fits
        rabi_freqfit = np.zeros(shape=(2,len(omega_vec)))
        for k in range(len(omega_vec)):
            omega_temp = omega_vec[k]
            rabi_freqfit[0, k] = power_wrt_omega(omega_temp, 0, rabi_data, self.time_stamps)
            rabi_freqfit[1, k] = power_wrt_omega(omega_temp, 1, rabi_data, self.time_stamps)

        rabi_freq = np.zeros(2)

        if omega_vec[0] == 0:
            ind_start = 1
        else:
            ind_start = 0

        if prior_freq is None:
            rabi_freq[0] = omega_vec[np.argmax(rabi_freqfit[0, ind_start:int((rabi_freqfit.shape[1] - 1) / 2)]) + ind_start]
            rabi_freq[1] = omega_vec[np.argmax(rabi_freqfit[1, ind_start:int((rabi_freqfit.shape[1] - 1) / 2)]) + ind_start]
        else:
            rabi_freq[0] = omega_vec[np.argmax(rabi_freqfit[0, ind_start:]) + ind_start]
            rabi_freq[1] = omega_vec[np.argmax(rabi_freqfit[1, ind_start:]) + ind_start]

        return omega_vec, rabi_freqfit, rabi_freq

    def plot_power_spectrum_linear_regression(self, rabi_data=None,
                                              prior_freq=[0,5e8], omega_truth=None, do_plot=False, freq_resolution=4):
        if rabi_data is None:
            rabi_data = self.pvec_data

        n_t = len(self.time_stamps)
        omega_array = (self.freq_convert/freq_resolution)*np.arange(freq_resolution*n_t)

        omega_vec, rabi_chisq, rabi_freq = self.power_spectrum_linear_regression(omega_array, rabi_data=rabi_data,
                                                                                 prior_freq=prior_freq)

        if do_plot:
            # Plot the Power Spectrum
            plt.figure(2, figsize=(10, 7))
            freq_plot = np.copy(omega_vec)
            rabi_chisq_plot = np.copy(rabi_chisq)
            xlabel_plot = 'Frequency'

            # Plot chisq
            plt.semilogx(freq_plot, rabi_chisq_plot[0, :], 'bo-', label=r'Control in $|0 \rangle$')
            plt.semilogx(freq_plot, rabi_chisq_plot[1, :], 'r^-', label=r'Control in $|1 \rangle$')

            plt.axvline(x=rabi_freq[0], linestyle='--', c='b')
            plt.axvline(x=rabi_freq[1], linestyle='--', c='r')

            if omega_truth is not None:
                plt.axvline(x=omega_truth[0], linestyle='-', c='b')
                plt.axvline(x=omega_truth[1], linestyle='-', c='r')

            plt.ylabel('Power')
            plt.xlabel(xlabel_plot)
            plt.legend(loc='best')

            plt.show()

            print(rabi_freq)
            if omega_truth is not None:
                print([np.abs(rabi_freq[0] - omega_truth[0]) / omega_truth[0],
                       np.abs(rabi_freq[1] - omega_truth[1]) / omega_truth[1]])


        return {'omega_array': omega_vec, 'rabi_chisq': rabi_chisq, 'rabi_freq': rabi_freq}

    def plot_corrected_power_spectrum_check(self, prior_freq=[0,1e8], omega_truth=None):
        # Get Normal FFT + Higher Resolution FFT
        rabi_power0, rabi_chisq0, omega_array0 = self.corrected_power_spectrum(prior_freq=prior_freq)
        rabi_power1, rabi_chisq1, omega_array1 = self.corrected_power_spectrum_halfway(prior_freq=prior_freq)

        # Sorting the collected arrays
        omega_array = np.concatenate((omega_array0, omega_array1))
        rabi_power = np.concatenate((rabi_power0, rabi_power1), axis=1)
        rabi_chisq = np.concatenate((rabi_chisq0, rabi_chisq1), axis=1)

        omega_sorted = np.sort(omega_array)
        sort_omega_index = np.argsort(omega_array)
        rabi_power_sorted = rabi_power[:, sort_omega_index]
        rabi_chisq_sorted = rabi_chisq[:, sort_omega_index]

        # Get the power computed from the linear regression fits
        rabi_freqfit = np.zeros(shape=rabi_chisq_sorted.shape)
        for k in range(len(omega_sorted)):
            omega_temp = omega_sorted[k]
            rabi_freqfit[0, k] = power_wrt_omega(omega_temp, 0, self.pvec_data, self.time_stamps)
            rabi_freqfit[1, k] = power_wrt_omega(omega_temp, 1, self.pvec_data, self.time_stamps)

        # Plot the Power Spectrum
        plt.figure(2, figsize=(10, 10))
        freq_plot = np.copy(omega_sorted)
        rabi_power_plot = np.copy(rabi_power_sorted)
        rabi_chisq_plot = np.copy(rabi_chisq_sorted)
        xlabel_plot = 'Frequency'

        plt.subplot(211)
        # Plot chisq
        plt.semilogx(freq_plot, rabi_chisq_plot[0, :], 'bo-')
        plt.semilogx(freq_plot, rabi_chisq_plot[1, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        plt.ylabel('Normal Equations')
        plt.xlabel(xlabel_plot)

        plt.subplot(212)
        # Plot chisq
        plt.semilogx(freq_plot, rabi_freqfit[0, :], 'bo-')
        plt.semilogx(freq_plot, rabi_freqfit[1, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        plt.ylabel('power wrt omega')
        plt.xlabel(xlabel_plot)

        plt.show()

        if freq_plot[0] == 0:
            ind_start = 1
        else:
            ind_start = 0

        rabi_freq = np.array([
            freq_plot[np.argmax(rabi_chisq_plot[0, ind_start:int((rabi_chisq_plot.shape[1] - 1) / 2)]) + ind_start],
            freq_plot[np.argmax(rabi_chisq_plot[1, ind_start:int((rabi_chisq_plot.shape[1] - 1) / 2)]) + ind_start]])

        print(rabi_freq)
        if omega_truth is not None:
            print([np.abs(rabi_freq[0] - omega_truth[0]) / omega_truth[0],
                   np.abs(rabi_freq[1] - omega_truth[1]) / omega_truth[1]])

    def point_estimator_linear_regression(self, rabi_data=None, freq_resolution=4):
        """
        rabi_data: If you want to calculate the point estimate of rabi oscillations for something besides the default
        """
        if rabi_data is None:
            rabi_data = self.pvec_data

        # Take fourier transforms of the rabi oscillations
        results_freq = self.plot_power_spectrum_linear_regression(rabi_data=rabi_data,
                                                                  prior_freq=[0,5e8], do_plot=False,
                                                                  freq_resolution=freq_resolution)

        rabi_freq = results_freq['rabi_freq']

        # Get the coarse estimates of rabi_gpt
        rabi_freq, rabi_gpt = self.model_free_coarse_hamiltonian_parameters_estimator(rabi_freq)

        # Refine the estimate
        rabi_refined_gpt = self.model_free_fine_hamiltonian_parameters_estimator(rabi_freq, rabi_gpt)

        param_array = np.array([rabi_freq[0] / 2, 2 * (rabi_refined_gpt[0, 2] - np.pi / 4), rabi_refined_gpt[0, 1],
                                rabi_freq[1] / 2, 2 * (rabi_refined_gpt[1, 2] - np.pi / 4), rabi_refined_gpt[1, 1]])

        return param_array

    def uncertainty_estimator(self, n_MC=100, do_plot=False):
        """
        n_MC: number of MC samples to consider from the sampling noise to be added to the pvec_data
        """
        p1_data = (1-self.pvec_data)/2

        param_vec = np.zeros(shape=(n_MC,6), dtype=float)

        # One should be as usual
        param_vec[0, :] = self.point_estimator()

        for ind_MC in range(1,n_MC):
            # Generate noisy samples
            noise_temp = np.random.binomial(self.nshots_rabi_data, p1_data)
            rabi_data_temp = 1.0 - 2.0*(noise_temp/self.nshots_rabi_data)

            # compute the point estimator
            param_vec[ind_MC,:] = self.point_estimator(rabi_data=rabi_data_temp)

        if do_plot:
            # Plot the histogram over omegas
            fig = plt.figure(0, figsize=(12, 4))
            ax1 = fig.add_subplot(121)
            ax1.hist(param_vec[:,0]/(1e6), bins=int(n_MC / 4), c='b')
            ax1.set(xlabel=r'$\omega_0 (\times 10^6)$', ylabel='Count')

            ax2 = fig.add_subplot(122)
            ax2.hist(param_vec[:,3]/(1e6), bins=int(n_MC / 4), c='r')
            ax2.set(xlabel=r'$\omega_1 (\times 10^6)$', ylabel='Count')

        return param_vec

    def uncertainty_estimator_linear_regression(self, n_MC=100, freq_resolution=4, do_plot=False):
        """
        n_MC: number of MC samples to consider from the sampling noise to be added to the pvec_data
        """
        p1_data = (1-self.pvec_data)/2

        param_vec = np.zeros(shape=(n_MC,6), dtype=float)
        rabi_freq_vec = np.zeros(shape=(n_MC, 2), dtype=float)

        # Create array of rabi oscillations
        rabi_data_vec = []

        # One should be as usual
        rabi_data_vec.append(self.pvec_data)
        results_freq_temp = self.plot_power_spectrum_linear_regression(rabi_data=self.pvec_data,
                                                                       prior_freq=[0, 5e8], do_plot=False,
                                                                       freq_resolution=freq_resolution)
        rabi_freq_vec[0, :] = results_freq_temp['rabi_freq']

        for ind_MC in range(1,n_MC):
            # Generate noisy samples
            noise_temp = np.random.binomial(self.nshots_rabi_data, p1_data)
            rabi_data_temp = 1.0 - 2.0*(noise_temp/self.nshots_rabi_data)

            rabi_data_vec.append(rabi_data_temp)

            # compute the point estimator
            results_freq_temp = self.plot_power_spectrum_linear_regression(rabi_data=rabi_data_temp,
                                                                           prior_freq=[0, 5e8], do_plot=False,
                                                                           freq_resolution=freq_resolution)
            rabi_freq_vec[ind_MC,:] = results_freq_temp['rabi_freq']

            if do_plot:
                if np.mod(ind_MC, 5) == 0:
                    print('%d' % ind_MC)

        # Fit log-normal distribution to rabi_freq_vec
        mu0 = np.mean(np.log(rabi_freq_vec[:, 0] / 1e6))
        mu1 = np.mean(np.log(rabi_freq_vec[:, 1] / 1e6))
        sigma0 = np.std(np.log(rabi_freq_vec[:, 0] / 1e6))
        sigma1 = np.std(np.log(rabi_freq_vec[:, 1] / 1e6))

        # Sample from the above log-normal distributions
        rabi_freq_vec_samples = np.zeros(shape=(n_MC,2))
        rabi_freq_vec_samples[:,0] = 1e6*np.random.lognormal(mu0, sigma0, n_MC)
        rabi_freq_vec_samples[:,1] = 1e6*np.random.lognormal(mu1, sigma1, n_MC)

        # Carry out linear regression for each sample
        for ind_MC in range(n_MC):
            rabi_freq = rabi_freq_vec_samples[ind_MC,:]
            rabi_data_temp = rabi_data_vec[ind_MC]

            # Get the coarse estimates of rabi_gpt
            rabi_freq, rabi_gpt = self.model_free_coarse_hamiltonian_parameters_estimator(rabi_freq,
                                                                                          rabi_data=rabi_data_temp)

            # Refine the estimate
            rabi_refined_gpt = self.model_free_fine_hamiltonian_parameters_estimator(rabi_freq, rabi_gpt,
                                                                                     rabi_data=rabi_data_temp)

            param_vec[ind_MC,:] = np.array([rabi_freq[0] / 2, 2 * (rabi_refined_gpt[0, 2] - np.pi / 4), rabi_refined_gpt[0, 1],
                                            rabi_freq[1] / 2, 2 * (rabi_refined_gpt[1, 2] - np.pi / 4), rabi_refined_gpt[1, 1]])

            if do_plot:
                if np.mod(ind_MC, 5) == 0:
                    print('%d\n' % ind_MC)

        if do_plot:
            # Plot the histogram over omegas
            fig = plt.figure(0, figsize=(14, 4))
            ax1 = fig.add_subplot(121)
            ax1.hist(param_vec[:,0]/(1e6), color='skyblue')
            ax1.set(xlabel=r'$\omega_0 (\times 10^6)$', ylabel='Count')

            ax2 = fig.add_subplot(122)
            ax2.hist(param_vec[:,3]/(1e6), color='r')
            ax2.set(xlabel=r'$\omega_1 (\times 10^6)$', ylabel='Count')

        return param_vec

    def uncertainty_estimator_bayesian(self, n_MC=100, do_plot=False, param_truth=None, freq_resolution=8):
        """
        n_MC: number of MC samples to consider from the sampling noise to be added to the pvec_data
        """
        param_vec = np.zeros(shape=(n_MC,6), dtype=float)

        # Set up the parameters of the beta distribution
        alpha_rabi_data = self.n0shots_rabi_data + 1/2
        beta_rabi_data = (self.nshots_rabi_data - self.n0shots_rabi_data) + 1/2

        # One should be as usual
        if self.FLAG_fft_high_resolution:
            param_vec[0, :] = self.point_estimator_linear_regression(freq_resolution=freq_resolution)
        else:
            param_vec[0, :] = self.point_estimator()

        for ind_MC in range(1,n_MC):
            # Generate noisy samples
            noise_temp = np.random.beta(alpha_rabi_data, beta_rabi_data)

            rabi_data_temp = 2.0*noise_temp - 1

            # compute the point estimator
            if self.FLAG_fft_high_resolution:
                param_vec[ind_MC, :] = self.point_estimator_linear_regression(rabi_data=rabi_data_temp,freq_resolution=freq_resolution)
            else:
                param_vec[ind_MC,:] = self.point_estimator(rabi_data=rabi_data_temp)

        if do_plot:
            type_distribution_fit = 'log_normal'

            if type_distribution_fit is 'log_normal':
                mu0 = np.mean(np.log(param_vec[:, 0] / 1e6))
                mu1 = np.mean(np.log(param_vec[:, 3] / 1e6))
                sigma0 = np.std(np.log(param_vec[:, 0] / 1e6))
                sigma1 = np.std(np.log(param_vec[:, 3] / 1e6))

                x0 = np.linspace(scipy.stats.lognorm.ppf(0.001, sigma0, scale=np.exp(mu0)), scipy.stats.lognorm.ppf(0.999, sigma0, scale=np.exp(mu0)), 1000)
                x1 = np.linspace(scipy.stats.lognorm.ppf(0.001, sigma1, scale=np.exp(mu0)), scipy.stats.lognorm.ppf(0.999, sigma1, scale=np.exp(mu1)), 1000)

                y0 = scipy.stats.lognorm.pdf(x0, sigma0, scale=np.exp(mu0))
                y1 = scipy.stats.lognorm.pdf(x1, sigma1, scale=np.exp(mu1))

                print(np.exp(mu0))
                print(sigma0)
                print(np.exp(mu1))
                print(sigma1)
            else:
                mu0 = np.mean(param_vec[:, 0] / 1e6)
                mu1 = np.mean(param_vec[:, 3] / 1e6)
                sigma0 = np.std(param_vec[:, 0] / 1e6)
                sigma1 = np.std(param_vec[:, 3] / 1e6)

                x0 = np.linspace(mu0 - 3 * sigma0, mu0 + 3 * sigma0, 100)
                x1 = np.linspace(mu1 - 3 * sigma1, mu1 + 3 * sigma1, 100)

                y0 = scipy.stats.norm.pdf(x0, mu0, sigma0)
                y1 = scipy.stats.norm.pdf(x1, mu1, sigma1)

                print(mu0)
                print(sigma0)
                print(mu1)
                print(sigma1)

            fig = plt.figure(0, figsize=(14, 4))
            ax1 = fig.add_subplot(121)
            ax1.hist(param_vec[:,0]/(1e6), bins=int(n_MC / 4), color='b')
            ax1.set_xscale('log')
            ax1.plot(x0, y0, 'k')

            if param_truth is not None:
                plt.axvline(x=param_truth[0]/1e6, linestyle='--', c='k')

            ax1.set(xlabel=r'$\omega_0 (\times 10^6)$', ylabel='Count')

            ax2 = fig.add_subplot(122)
            ax2.hist(param_vec[:,3]/(1e6), bins=int(n_MC / 4), color='r')
            ax2.set_xscale('log')
            ax2.plot(x1, y1, 'k')

            if param_truth is not None:
                plt.axvline(x=param_truth[1]/1e6, linestyle='--', c='k')

            ax2.set(xlabel=r'$\omega_1 (\times 10^6)$', ylabel='Count')

            fig = plt.figure(1, figsize=(16, 4))
            ax1 = fig.add_subplot(121)
            ax1.plot(x0, y0, 'k')
            ax1.set_xscale('log')

            if param_truth is not None:
                plt.axvline(x=param_truth[0] / 1e6, linestyle='--', c='k')

            ax1.set(xlabel=r'$\omega_0 (\times 10^6)$')

            ax2 = fig.add_subplot(122)
            ax2.plot(x1, y1, 'k')
            ax2.set_xscale('log')

            if param_truth is not None:
                plt.axvline(x=param_truth[1] / 1e6, linestyle='--', c='k')

            ax2.set(xlabel=r'$\omega_1 (\times 10^6)$')

        return param_vec

    def param_distribution(self):
        '''
        Get the log-normal distribution for omega's or the main different values of omega
        '''
    def line_search_estimator(self, prep, FLAG_mle_solve=False, verbose=False, FLAG_debug=False, do_plot=False):
        # Consider a range of frequencies
        N_t = len(self.time_stamps)
        set_frequencies = np.array([ind*(self.sample_freq/(2*N_t)) for ind in range(N_t + 1)])
        n_freqs = len(set_frequencies)

        # For each value of frequency, get an estimate of the rabi amplitudes A, B, C based on regression
        set_rabi_abc = []
        for freq in set_frequencies:
            _, rabi_abc_temp = self.model_free_coarse_rabi_amplitude_estimator(freq, prep, FLAG_refine_rabi_freq=False)
            set_rabi_abc.extend([rabi_abc_temp])

        # Use these as seed values to MLE estimator to get finer estimates of A, B and C
        config_array = np.array([0, 2, 4]) + prep

        if FLAG_mle_solve:
            for ind_freq in range(n_freqs):
                if FLAG_debug:
                    print(ind_freq)

                counter = 0
                for ind_config in config_array:
                    ds = self.rabi_amplitude_mle_estimator(set_rabi_abc[ind_freq][counter,:],
                                                           set_frequencies[ind_freq],
                                                           ind_config, verbose=verbose, do_plot=do_plot)

                    set_rabi_abc[ind_freq][counter,:] = ds['results'][1]

        # Obtain a probability distribution over set of discrete frequencies
        p_freq = np.zeros(n_freqs)
        for ind_freq in range(n_freqs):
            log_temp_freq = 1
            counter = 0
            for ind_config in config_array:
                p0_temp = np_model_free_probabilities(set_rabi_abc[ind_freq][counter,:],
                                                      set_frequencies[ind_freq],
                                                      self.tvec_config[ind_config])

                Y_temp = self.samples_config[ind_config]
                log_p_samples_temp = np.sum(np.log( (1-Y_temp)*p0_temp + Y_temp*(1 - p0_temp) + 1e-8 ))

                log_temp_freq += log_p_samples_temp
                counter += 1
                print('ind_freq: %d, counter: %d, config: %d, log_p_samples: %f' %(ind_freq, counter, ind_config, log_p_samples_temp))

                # import pdb; pdb.set_trace()
            p_freq[ind_freq] = np.exp(log_temp_freq)

        p_freq = p_freq/(np.sum(p_freq) + 1e-8)

        # Sample from this set and run the rabi amplitude estimator for each value

        return set_frequencies, p_freq, set_rabi_abc


class PointEstimator(object):
    """
    Use rabi oscillations for estimation of Hamiltonian parameters
    1. Primarily uses functions from "fft_estimators" and "linear_estimators"
    2. Doesn't incorporate uncertainty

    TODO: Add options for different parameterizations
    """

    def __init__(self, A_cr, do_plot=False,
                 FLAG_fft_window=False, type_fft_window=None,
                 FLAG_fft_high_resoultion=False,
                 FLAG_verbose=False):

        # Regarding the use of FFT
        self.FLAG_fft_window = FLAG_fft_window
        self.type_fft_window = type_fft_window
        self.FLAG_fft_high_resolution = FLAG_fft_high_resoultion

        # Extract relevant parameters
        time_stamps = A_cr.tset * A_cr.xi_t
        freq_convert = A_cr.freq_convert

        # Calculate the rabi oscillations from the given data
        pvec_data = A_cr.rabi_data()

        self.time_stamps = time_stamps
        self.pvec_data = pvec_data
        self.sample_freq = A_cr.sample_freq
        self.freq_convert = freq_convert

        # Other options
        self.do_plot = do_plot
        self.FLAG_verbose = FLAG_verbose

    def model_free_fine_rabi_frequency_estimator(self, rabi_freq, prep, rabi_data=None):
        """
        Given a value of omega for a particular prep op in A cos(wt) + B sin(wt) + C, refine this based on regression
        """
        # Calculate and store the initial frequency fit considering this value of omega for each of X, Y, Z meas ops

        if rabi_data is None:
            rabi_data = self.pvec_data

        rabi_prefit = np.ndarray(shape=(3, rabi_data.shape[1]), dtype=float)

        config_array = np.array([0, 2, 4]) + prep
        counter = 0
        for ind_config in config_array:
            cs_array = fft_estimators.cos_sin_features(rabi_freq, self.time_stamps)
            reg = LinearRegression().fit(cs_array, rabi_data[ind_config, :])
            rabi_prefit[counter] = reg.predict(cs_array)
            counter += 1

        # Refine the initial frequency estimates by fitting to the Rabi Oscillations
        ## INTRODUCE TRY AND CATCH WITH 2*freq_convert below
        bracketing_factors = np.arange(1,4)
        n_fails = 0
        while True:
            try:
                bracketing_half_width = bracketing_factors[n_fails]*self.freq_convert
                res = minimize_scalar(fft_estimators.sumsq_wrt_omega,
                                      bracket=(rabi_freq - bracketing_half_width,
                                               rabi_freq,
                                               rabi_freq + bracketing_half_width),
                                      args=(prep, rabi_data, self.time_stamps))
                break
            except ValueError:
                n_fails += 1

                if n_fails > 2:
                    print('Max no. of trials reached')
                    return None
                else:
                    print('ValueError, trying again with different backeting interval')

        return res.x

    def model_free_coarse_rabi_amplitude_estimator(self, rabi_freq, prep, rabi_data=None, FLAG_refine_rabi_freq=False):
        """
        Estimate the amplitudes i.e., A, B and C given particular values of omega in
        A cos(wt) + B sin(wt) + C

        Inputs:
        rabi_freq - scalar quantity here
        prep - index of the state preparation operator where U0 = II and U1 = XI, takes values 0 or 1
        """
        # Refine initial frequency estimates
        if rabi_data is None:
            rabi_data = self.pvec_data

        if FLAG_refine_rabi_freq:
            rabi_freq_refined = self.model_free_fine_rabi_frequency_estimator(rabi_freq, prep, rabi_data=rabi_data)

            # Indicative of error!
            if rabi_freq_refined is not None:
                rabi_freq = rabi_freq_refined

        config_array = np.array([0,2,4]) + prep

        # Calculate and store the fitted curves and corresponding Fourier coefficients
        rabi_cos_sin = np.ndarray(shape=(3,rabi_data.shape[1]), dtype=float)
        rabi_residuals = np.ndarray(shape=(3,rabi_data.shape[1]), dtype=float)
        rabi_abc = np.ndarray(shape=(3, 3), dtype=float)

        counter = 0
        for ind_config in config_array:
            cs_array = fft_estimators.cos_sin_features(rabi_freq, self.time_stamps)
            reg = LinearRegression().fit(cs_array, rabi_data[ind_config, :])
            rabi_abc[counter, 0] = reg.coef_[0]
            rabi_abc[counter, 1] = reg.coef_[1]
            rabi_abc[counter, 2] = reg.intercept_
            rabi_cos_sin[counter] = reg.predict(cs_array)
            rabi_residuals[counter] = rabi_data[ind_config, :] - rabi_cos_sin[counter]
            counter += 1

        return rabi_freq, rabi_abc

    def convert_rabi_amplitudes_to_hamiltonian_parameters(self, rabi_abc):
        """
        Given amplitudes A, B, C of rabi oscillations, obtain the hamiltonian parameters (as in the spectral param)

        rabi_abc is an array of amplitudes A, B, C for the different meas ops of X, Y, Z but only for a given prep op
        """
        # Calculate initial guesses for gamma, phi, theta
        rabi_gpt = np.ndarray(shape=(3,), dtype=float)

        g0 = -cmath.phase(complex(rabi_abc[2, 0], rabi_abc[2, 1]))
        expg0 = cmath.exp(complex(0, g0))
        x0 = expg0 * complex(rabi_abc[0, 0], rabi_abc[0, 1])
        y0 = expg0 * complex(rabi_abc[1, 0], rabi_abc[1, 1])
        z0 = expg0 * complex(rabi_abc[2, 0], rabi_abc[2, 1])
        rabi_gpt[0] = g0
        rabi_gpt[1] = cmath.phase(complex(-y0.imag, x0.imag))

        if z0.real > 1.0:
            rabi_gpt[2] = 0.25 * np.pi
        else:
            rabi_gpt[2] = (0.25 * np.pi) + 0.5 * np.sign(x0.real * y0.imag) * math.acos(z0.real)

        return rabi_gpt

    def model_free_coarse_hamiltonian_parameters_estimator(self, rabi_freq_array, rabi_data=None):
        if rabi_data is None:
            rabi_data = self.pvec_data

        # Get rabi amplitude estimates for each prep op
        rabi_freq_U0, rabi_abc_U0 = self.model_free_coarse_rabi_amplitude_estimator(rabi_freq_array[0], 0,
                                                                                    rabi_data=rabi_data,
                                                                                    FLAG_refine_rabi_freq=False)

        rabi_freq_U1, rabi_abc_U1 = self.model_free_coarse_rabi_amplitude_estimator(rabi_freq_array[1], 1,
                                                                                    rabi_data=rabi_data,
                                                                                    FLAG_refine_rabi_freq=False)

        # Convert to hamiltonian parameters (spectral param)
        rabi_gpt = np.ndarray(shape=(2, 3), dtype=float)
        rabi_gpt[0,:] = self.convert_rabi_amplitudes_to_hamiltonian_parameters(rabi_abc_U0)
        rabi_gpt[1,:] = self.convert_rabi_amplitudes_to_hamiltonian_parameters(rabi_abc_U1)

        # Return refined values of rabi_freq
        rabi_freq = np.array([rabi_freq_U0, rabi_freq_U1])

        return rabi_freq, rabi_gpt

    def model_free_fine_hamiltonian_parameters_estimator(self, rabi_freq, rabi_gpt, rabi_data=None):
        if rabi_data is None:
            rabi_data = self.pvec_data

        rabi_refined_gpt = np.ndarray(shape=(2, 3), dtype=float)

        res = minimize(linear_estimators.sumsq_wrt_gpt, rabi_gpt[0, :],
                       args=(0, rabi_data, rabi_freq, self.time_stamps),
                       method="trust-constr",
                       jac=False,  # grad_sumsq_wrt_gpt,
                       bounds=((-math.pi, math.pi),
                               (-math.pi, math.pi),
                               (0, 0.5 * math.pi))
                       )

        rabi_refined_gpt[0, :] = res.x

        res = minimize(linear_estimators.sumsq_wrt_gpt, rabi_gpt[1, :],
                       args=(1, rabi_data, rabi_freq, self.time_stamps),
                       method="trust-constr",
                       jac=False,  # grad_sumsq_wrt_gpt,
                       bounds=((-math.pi, math.pi),
                               (-math.pi, math.pi),
                               (0, 0.5 * math.pi))
                       )
        rabi_refined_gpt[1, :] = res.x

        return rabi_refined_gpt

    def baseline(self, rabi_data=None):
        """
        Baseline solver -- Doesn't refine the estimate of frequencies after FFT

        So basically FFT + Regression (as advertised in the paper)

        Inputs:
            rabi_data: rabi oscillations if not the default
        """
        if rabi_data is None:
            rabi_data = self.pvec_data

        # Take fourier transforms of the rabi oscillations
        results_freq = fft_estimators.calc_freq(rabi_data, self.freq_convert, sample_freq=self.sample_freq,
                                                do_plot=self.do_plot,
                                                FLAG_window=self.FLAG_fft_window, type_window=self.type_fft_window,
                                                FLAG_high_resolution=self.FLAG_fft_high_resolution)

        rabi_freq = results_freq['rabi_freq']

        if self.FLAG_verbose:
            print('FFT Frequency Estimates')
            print(rabi_freq)

        # Get the coarse estimates of rabi_gpt
        rabi_freq, rabi_gpt = self.model_free_coarse_hamiltonian_parameters_estimator(rabi_freq)

        # Refine the estimate
        rabi_refined_gpt = self.model_free_fine_hamiltonian_parameters_estimator(rabi_freq, rabi_gpt)

        if self.FLAG_verbose:
            print('Linear Regression Estimates')
            print(rabi_freq)

        param_array = np.array([rabi_freq[0] / 2, 2 * (rabi_refined_gpt[0, 2] - np.pi / 4), rabi_refined_gpt[0, 1],
                                rabi_freq[1] / 2, 2 * (rabi_refined_gpt[1, 2] - np.pi / 4), rabi_refined_gpt[1, 1]])

        return param_array

    def point_estimator(self, rabi_data=None):
        """
        Baseline solver and a bit more -- frequencies after FFT are refined using a regression step

        Inputs:
            rabi_data: rabi oscillations if not the default
        """
        if rabi_data is None:
            rabi_data = self.pvec_data

        # Take fourier transforms of the rabi oscillations
        results_freq = fft_estimators.calc_freq(rabi_data, self.freq_convert, sample_freq=self.sample_freq,
                                                do_plot=self.do_plot,
                                                FLAG_window=self.FLAG_fft_window, type_window=self.type_fft_window,
                                                FLAG_high_resolution=self.FLAG_fft_high_resolution)

        rabi_freq = results_freq['rabi_freq']

        if self.FLAG_verbose:
            print('FFT Frequency Estimates')
            print(rabi_freq)

        # Refine the estimate of frequencies
        for prep_ind in [0, 1]:
            rabi_freq[prep_ind] = self.model_free_fine_rabi_frequency_estimator(rabi_freq[prep_ind], prep_ind)

        if self.FLAG_verbose:
            print('FFT Frequency Estimates')
            print(rabi_freq)

        # Get the coarse estimates of rabi_gpt
        rabi_freq, rabi_gpt = self.model_free_coarse_hamiltonian_parameters_estimator(rabi_freq)

        # Refine the estimate
        rabi_refined_gpt = self.model_free_fine_hamiltonian_parameters_estimator(rabi_freq, rabi_gpt)

        if self.FLAG_verbose:
            print('Linear Regression Estimates')
            print(rabi_freq)

        param_array = np.array([rabi_freq[0] / 2, 2 * (rabi_refined_gpt[0, 2] - np.pi / 4), rabi_refined_gpt[0, 1],
                                rabi_freq[1] / 2, 2 * (rabi_refined_gpt[1, 2] - np.pi / 4), rabi_refined_gpt[1, 1]])

        return param_array
