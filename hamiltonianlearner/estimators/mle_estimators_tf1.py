#
# Contains SGD methods to solve the MLE of Hamiltonian Learning using TensorFlow v1
# Kept for historical reasons (will be removed in future!)
#

import numpy as np
import math
import cmath
from absl import flags
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

import matplotlib.pyplot as plt
import functools

from ..quantum_system_oracles import simulate_nature
from ..quantum_system_models import quantum_device_models

# Set number of threads of tensorflow
tf_num_threads=1
tf.config.threading.set_inter_op_parallelism_threads(tf_num_threads)


# MLE Estimator
class MLE_Estimator(object):
    '''
    Sets up the MLE Estimator with different solver options and loss functions

    TODO: Add options for different parameterizations and loss functions
    '''

    def __init__(self, data, xi_param, solver_options=None):
        '''
        A general setup that doesn't assume we use tensorflow for SGD and hence additional methods can
        be added without modifying this initialization
        '''
        self.samples = np.array(data['samples']).astype(np.float32)
        self.mvec = np.array(data['mvec']).astype(np.int16)
        self.uvec = np.array(data['uvec']).astype(np.int16)
        self.tvec = np.array(data['tvec']).astype(np.float32)
        self.xi_param = np.array(xi_param).astype(np.float32)

        # One part of the loss function is decided if the data is classified or GMM fitting was used
        if data['FLAG_classification'] is False:
            # Probability of a signal given readout was 0
            self.p_c_given_meas_0 = np.array(data['samples_p0']).astype(np.float32)

            # Probability of a signal given readout was 1
            self.p_c_given_meas_1 = np.array(data['samples_p1']).astype(np.float32)
        elif data['FLAG_classification'] is True:
            self.ybin = data['samples']
            self.r0, self.r1 = data['misclassif_error']
        else:
            raise RuntimeError('Cannot continue with current data[FLAG_classification]!')

        self.FLAG_classification = data['FLAG_classification']

        # Create weight matrices based on mvec
        self.w0_vec = np.zeros(self.mvec.size).astype(np.int16)
        self.w1_vec = np.zeros(self.mvec.size).astype(np.int16)
        self.w2_vec = np.zeros(self.mvec.size).astype(np.int16)

        self.w0_vec[self.mvec == 0] = 1
        self.w1_vec[self.mvec == 1] = 1
        self.w2_vec[self.mvec == 2] = 1

        # Setup solver options
        default_solver_options = {'nepochs': 200, 'neval_period': 10,
                                  'learning_rate': 0.001, 'optimizer': 'adam',
                                  'mini_batch_size': 512}

        if solver_options is None:
            self.nepochs = default_solver_options['nepochs']
            self.neval_period = default_solver_options['neval_period']
            self.learning_rate = default_solver_options['learning_rate']
            self.optimizer = default_solver_options['optimizer']
            self.mini_batch_size = default_solver_options['mini_batch_size']
        else:
            if 'nepochs' in solver_options.keys():
                self.nepochs = solver_options['nepochs']
            else:
                self.nepochs = default_solver_options['nepochs']

            if 'neval_period' in solver_options.keys():
                self.neval_period = solver_options['neval_period']
            else:
                self.neval_period = default_solver_options['neval_period']

            if 'learning_rate' in solver_options.keys():
                self.learning_rate = solver_options['learning_rate']
            else:
                self.learning_rate = default_solver_options['learning_rate']

            if 'optimizer' in solver_options.keys():
                self.optimizer = solver_options['optimizer']
            else:
                self.optimizer = default_solver_options['optimizer']

            if 'mini_batch_size' in solver_options.keys():
                self.mini_batch_size = solver_options['mini_batch_size']
            else:
                self.mini_batch_size = default_solver_options['mini_batch_size']

        # Decide on the control noise model to be used -- this needs to be updated
        if 'device' in data.keys():
            if data['device'] == 'ibmq_boeblingen':
                self.FLAG_boel = True
        else:
            self.FLAG_boel = False

    def np_loss(self, param_num, type_param='Theta', verbose=True):
        # model variables
        param_nd = param_num/self.xi_param

        # model to predict sample probability (prob(k=0))
        if self.FLAG_boel:
            if verbose:
                print('Using ibmq_boel device data-driven teff model for %s' % type_param)

            Y_sp = model_probabilities_ibmq_np_nd(param_nd, type_param,
                                                  self.xi_param, self.uvec,
                                                  self.w0_vec, self.w1_vec,
                                                  self.w2_vec,self.tvec)

        # loss function (binary cross-entropy)
        Y = self.samples
        loss = -1 * np.log((1 - Y) * ((1 - self.r0) * Y_sp + self.r1 * (1 - Y_sp)) + Y * (
                (1 - self.r1) * (1 - Y_sp) + self.r0 * Y_sp) + 1.0e-10)  # log likelihood
        loss = np.mean(loss)  # take mean over batch

        return loss


    def tf_loss(self, param_num, type_param='Theta', verbose=True):
        # graph input
        T = tf.placeholder(tf.float32, name="X_time_points")
        W0 = tf.placeholder(tf.float32, name="X_w0_meas_operators")
        W1 = tf.placeholder(tf.float32, name="X_w1_meas_operators")
        W2 = tf.placeholder(tf.float32, name="X_w2_meas_operators")
        U = tf.placeholder(tf.float32, name="X_prep_operators")
        Y = tf.placeholder(tf.float32, name="Y_samples")  # y = 0 or 1
        XI = tf.placeholder(tf.float32, name="XI_param")  # y = 0 or 1

        # model variables
        param_nd = param_num/self.xi_param
        theta_nd = tf.Variable(param_nd, name="theta_nd", dtype=tf.float32)

        # model to predict sample probability (prob(k=0))
        if self.FLAG_boel:
            if verbose:
                print('Using ibmq_boel device data-driven teff model for %s' % type_param)

            if type_param == 'Theta':
                Y_sp = model_probabilities_param_ibmq_tf_nd(theta_nd, XI, U, W0, W1, W2, T)
            elif type_param == 'J':
                Y_sp = model_probabilities_J_ibmq_tf_nd(theta_nd, XI, U, W0, W1, W2, T)
            else:
                raise RuntimeError('Passed wrong parameterization!')
        else:
            if verbose:
                print('Using old ibmq device data-driven teff model')

            if type_param == 'Theta':
                Y_sp = model_probabilities_param_tf_nd(theta_nd, XI, U, W0, W1, W2, T)
            elif type_param == 'J':
                Y_sp = model_probabilities_J_tf_nd(theta_nd, XI, U, W0, W1, W2, T)
            else:
                raise RuntimeError('Passed wrong parameterization!')

        # loss function (binary cross-entropy)
        loss = -1 * tf.log((1 - Y) * ((1 - self.r0) * Y_sp + self.r1 * (1 - Y_sp)) + Y * (
                (1 - self.r1) * (1 - Y_sp) + self.r0 * Y_sp) + 1.0e-10)  # log likelihood
        loss = tf.reduce_mean(loss)  # take mean over batch

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            results = sess.run([loss, theta_nd],
                               feed_dict={T: self.tvec,
                                          U: self.uvec,
                                          W0: self.w0_vec,
                                          W1: self.w1_vec,
                                          W2: self.w2_vec,
                                          Y: self.samples,
                                          XI: self.xi_param})

        return results

    def tf_sgd(self, init_param_nd, type_param='Theta', verbose=True, do_plot=False):
        '''
        Generalization for mle_estimation_param for now

        and works only with parameterization of Lambda and assumes binary classifier
        '''
        if verbose:
            print("Model training for %s epochs, with evaluation every %s steps" % (self.nepochs, self.neval_period))

        # Extract solver options
        nepochs = self.nepochs
        neval_period = self.neval_period
        learning_rate = self.learning_rate
        mini_batch_size = self.mini_batch_size

        batch_size = len(self.tvec)    # MLE Batch Size

        if self.FLAG_classification:
            r0 = tf.constant(self.r0, dtype=tf.float32)
            r1 = tf.constant(self.r1, dtype=tf.float32)

        xy = np.stack([self.tvec, self.samples, self.uvec, self.w0_vec, self.w1_vec, self.w2_vec], axis=1)

        n_mini_batches = int(xy.shape[0] / mini_batch_size)

        # graph input
        T = tf.placeholder(tf.float32, name="X_time_points")
        W0 = tf.placeholder(tf.float32, name="X_w0_meas_operators")
        W1 = tf.placeholder(tf.float32, name="X_w1_meas_operators")
        W2 = tf.placeholder(tf.float32, name="X_w2_meas_operators")
        U = tf.placeholder(tf.float32, name="X_prep_operators")
        Y = tf.placeholder(tf.float32, name="Y_samples")  # y = 0 or 1
        XI = tf.placeholder(tf.float32, name="XI_param")  # y = 0 or 1

        # model variables
        theta_nd = tf.Variable(init_param_nd, name="theta_nd", dtype=tf.float32)

        if verbose:
            print("Input data xy shape=%s" % str(xy.shape))
            print("Each epoch has %d mini_batches of size %s" % (n_mini_batches, mini_batch_size))
            print("Input data shapes samples=%s, tpts=%s" % (self.samples.shape, self.tvec.shape))
            print("Input data placeholders X=%s, W0=%s, W1=%s, W2=%s, U=%s, Y=%s, XI=%s" % (T, W0, W1, W2, U, Y, XI))
            print("initial values: theta=%s" % (init_param_nd,))

        # model to predict sample probability (prob(k=0))
        if self.FLAG_boel:
            if verbose:
                print('Using ibmq_boel device data-driven teff model for %s' % type_param)

            if type_param == 'Theta':
                Y_sp = model_probabilities_param_ibmq_tf_nd(theta_nd, XI, U, W0, W1, W2, T)
            elif type_param == 'J':
                Y_sp = model_probabilities_J_ibmq_tf_nd(theta_nd, XI, U, W0, W1, W2, T)
            else:
                raise RuntimeError('Passed wrong parameterization!')
        else:
            if verbose:
                print('Using old ibmq device data-driven teff model')

            if type_param == 'Theta':
                Y_sp = model_probabilities_param_tf_nd(theta_nd, XI, U, W0, W1, W2, T)
            elif type_param == 'J':
                Y_sp = model_probabilities_J_tf_nd(theta_nd, XI, U, W0, W1, W2, T)
            else:
                raise RuntimeError('Passed wrong parameterization!')

        # loss function (binary cross-entropy)
        loss = -1 * tf.log((1 - Y) * ((1 - r0) * Y_sp + r1 * (1 - Y_sp)) + Y * (
                (1 - r1) * (1 - Y_sp) + r0 * Y_sp) + 1.0e-10)  # log likelihood
        loss = tf.reduce_mean(loss)  # take mean over batch

        # optimizer
        if self.optimizer == "gd":
            gdo = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optimizer_op = gdo.minimize(loss)
            op_name = "GD"
        elif self.optimizer == "adagrad":
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
                                                      U: xy[n0:n0 + mini_batch_size, 2],
                                                      W0: xy[n0:n0 + mini_batch_size, 3],
                                                      W1: xy[n0:n0 + mini_batch_size, 4],
                                                      W2: xy[n0:n0 + mini_batch_size, 5],
                                                      XI: self.xi_param})

                if not (k % neval_period):
                    results = sess.run([loss, theta_nd],
                                       feed_dict={T: self.tvec,
                                                  U: self.uvec,
                                                  W0: self.w0_vec,
                                                  W1: self.w1_vec,
                                                  W2: self.w2_vec,
                                                  Y: self.samples,
                                                  XI: self.xi_param})
                    if verbose:
                        print("    Epoch %s: loss=%s, theta=%s" % tuple([k] + results))
                    losses.append(results[0])
                    steps.append(k)
                    if np.isnan(results[0]):
                        raise Exception("loss is NaN, quitting!")

            results = sess.run([loss, theta_nd],
                               feed_dict={T: self.tvec,
                                          U: self.uvec,
                                          W0: self.w0_vec,
                                          W1: self.w1_vec,
                                          W2: self.w2_vec,
                                          Y: self.samples,
                                          XI: self.xi_param})

        m_loss, m_theta = results

        # # Need to fix this!
        # if verbose:
        #     print("Results from ML regression: loss=%s, theta=%s" % (m_loss, m_J))
        #     if 'Jix' in data:
        #         J_num = np.array([data['Jix'], data['Jiy'], data['Jzx']])
        #         rmse = np.linalg.norm(m_J - J_num)
        #         print("Actual Jix=%s, Jiy=%s, Jzx=%s, rmse=%s" % (data['Jix'], data['Jiy'], data['Jzx'], rmse))

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

    def energy_landscape(self, param_num, type_param='J',
                         FLAG_readout_noise=True,
                         FLAG_imperfect_pulses=True, type_imperfect_pulse='constant', teff=[0, 0],
                         FLAG_loglikelihood_loss=True, FLAG_return=False, FLAG_ibmq_boel=False,
                         do_plot=True, title_plot='optimization_landscape_cr.eps'
                         ):

        # Considering different noise models when calculating the probabilities
        if FLAG_readout_noise is True:
            if self.FLAG_classification:
                type_readout_noise = 'classifier'
                r0 = self.r0
                r1 = self.r1
            else:
                type_readout_noise = 'gmm'
                # Already included in the above conditional probability description
        else:
            # Should not consider any readout noise model
            type_readout_noise = 'classifier'
            r0 = 0.0
            r1 = 0.0

        # Imperfect pulse-shaping model
        if FLAG_imperfect_pulses is True:
            if type_imperfect_pulse == 'none':
                # tvec remains as it is
                teff = [0, 0]
                tvec_expt = np.copy(self.tvec)
            elif type_imperfect_pulse == 'constant':
                tvec_expt = self.tvec + (1 - uvec) * teff[0] + uvec * teff[1]
            elif type_imperfect_pulse == 'data_driven':
                # Define the function (make this passable later)
                delta_t_imperfect_pulse = lambda omega, coeff0, coeff1: coeff0 / (omega + coeff1 * omega ** 2)
        elif FLAG_imperfect_pulses is False:
            tvec_expt = np.copy(self.tvec)

        # Parameters independent of if whether 'J' or 'Theta'
        param_nd = param_num / self.xi_param
        n_param = 150  # number of points along each slice
        d_param = 0.5  # spacing taking xi_param into account

        ## Create slices for the parameters according to the parameterization and relevant constraints
        # (6 rows corresponding to 6 slices, columns contain values for different parameter components)
        llvec = np.zeros((6, n_param))
        param_vec = np.zeros((6, n_param))

        if type_param == 'J':
            for ind_slice in range(6):
                param_pq = param_nd[ind_slice]
                param_slice = np.linspace(param_pq - d_param, param_pq + d_param, n_param)
                param_vec[ind_slice, :] = param_slice

                # Function to calculate the probabilities for log-likelihood
                compute_model_prob = lambda param, t: model_probabilities_np(param, self.uvec,
                                                                             self.w0_vec, self.w1_vec, self.w2_vec, t)

                # For plotting
                xlabel_array = [r'$J_{IX}$ ($10^8$ Hz)', r'$J_{IY}$ ($10^8$ Hz)', r'$J_{IZ}$ ($10^8$ Hz)',
                                r'$J_{ZX}$ ($10^8$ Hz)', r'$J_{ZY}$ ($10^8$ Hz)', r'$J_{ZZ}$ ($10^8$ Hz)']

        elif type_param == 'Theta':
            for ind_slice in range(6):
                param_pq = param_nd[ind_slice]

                if ind_slice in [0, 3]:
                    # parameter of interest is omega
                    param_vec_slice = np.linspace(np.amax([param_pq - d_param, 0]), param_pq + d_param, n_param)
                elif ind_slice in [1, 4]:
                    # parameter of interest is delta
                    param_vec_slice = np.linspace(-np.pi / 2, np.pi / 2, n_param)
                elif ind_slice in [2, 5]:
                    # parameter of interest is phi
                    param_vec_slice = np.linspace(-np.pi, np.pi, n_param)

                param_vec[ind_slice, :] = param_vec_slice

                compute_model_prob = lambda param, t: model_probabilities_param_np(param, self.uvec,
                                                                                   self.w0_vec, self.w1_vec,
                                                                                   self.w2_vec, t)

                # For plotting
                xlabel_array = [r'$\omega_0$ ($10^7$ Hz)', r'$\delta_0$ (radians)', r'$\phi_0$ ($10^{-1}$ radians)',
                                r'$\omega_1$ ($10^8$ Hz)', r'$\delta_1$ (radians)', r'$\phi_1$ ($10$ radians)']

        # Calculate the energy landscapes along each slice
        for ind_slice in range(6):
            param_vec_slice = param_vec[ind_slice, :]

            counter = 0
            for param_pq_temp in param_vec_slice:
                param_temp = np.copy(param_nd)
                param_temp[ind_slice] = param_pq_temp
                param_temp = param_temp * self.xi_param

                # Readout noise model has been defined and will be taken care of below

                # Imperfect pulse shaping
                if FLAG_imperfect_pulses is True:
                    if type_imperfect_pulse == 'data_driven':
                        # Calculate teff0 and teff1 using the curve fits
                        if type_param == 'J':
                            param_array_temp = quantum_device_models.transform_parameters(param_temp)
                            teff0, teff1 = quantum_device_models.data_driven_teff_noise_model(param_array_temp,
                                                                        FLAG_ibmq_boel=FLAG_ibmq_boel)
                        elif type_param == 'Theta':
                            teff0, teff1 = quantum_device_models.data_driven_teff_noise_model(param_temp, FLAG_ibmq_boel=FLAG_ibmq_boel)

                        tvec_expt = self.tvec + (1 - self.uvec) * teff0 + self.uvec * teff1

                Y_sp = compute_model_prob(param_temp, tvec_expt)

                if FLAG_loglikelihood_loss:
                    if type_readout_noise == 'gmm':
                        loss = -1 * np.log(self.p_c_given_meas_0 * Y_sp + self.p_c_given_meas_1 * (1 - Y_sp))
                    elif type_readout_noise == 'classifier':
                        loss = -1 * np.log((1 - self.ybin) * ((1 - self.r0) * Y_sp +
                                                self.r1 * (1 - Y_sp)) +
                                                self.ybin * ((1 - self.r1) * (1 - Y_sp) + self.r0 * Y_sp) + 1.0e-10)  # log likelihood

                    llvec[ind_slice, counter] = loss.mean()
                else:
                    if type_readout_noise == 'gmm':
                        ll = np.sum(self.p_c_given_meas_0 * Y_sp + self.p_c_given_meas_1 * (1 - Y_sp))
                    elif type_readout_noise == 'classifier':
                        ll = np.sum((1 - self.ybin) * ((1 - self.r0) * Y_sp + self.r1 * (1 - Y_sp)) +
                                    self.ybin * ((1 - self.r1) * (1 - Y_sp) + self.r0 * Y_sp) + 1.0e-10)

                    llvec[ind_slice, counter] = -1 * np.log(ll / len(ybin))

                counter += 1

        if do_plot:
            # Convert into Hz for plotting purposes (angular frequencies are returned though)
            if type_param == 'J':
                param_vec_plot = param_vec/(2*np.pi)
                param_nd_plot = param_nd/(2*np.pi)
            elif type_param == 'Theta':
                param_vec_plot = np.copy(param_vec)
                param_vec_plot[0] = param_vec_plot[0] / (2 * np.pi)
                param_vec_plot[3] = param_vec_plot[3] / (2 * np.pi)

                param_nd_plot = np.copy(param_nd)
                param_nd_plot[0] = param_nd_plot[0] / (2 * np.pi)
                param_nd_plot[3] = param_nd_plot[3] / (2 * np.pi)

            plt.figure(3, figsize=(12, 18))
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = "sans-serif"
            plt.rcParams['font.sans-serif'] = "Helvetica"
            plt.rcParams['figure.titlesize'] = 24
            plt.rcParams['axes.labelsize'] = 24
            plt.rcParams['axes.titlesize'] = 24
            plt.rcParams['legend.fontsize'] = 24
            plt.rcParams['xtick.labelsize'] = 24
            plt.rcParams['ytick.labelsize'] = 24

            subplot_indices = [321, 323, 325, 322, 324, 326]

            for ind in range(6):
                plt.subplot(subplot_indices[ind])
                plt.plot(param_vec_plot[ind, :], llvec[ind, :], linewidth=3)
                plt.axvline(x=param_nd_plot[ind], c='r', linestyle='--', linewidth=3)
                print(param_nd[ind])
                plt.grid(True)
                plt.xlabel(xlabel_array[ind], labelpad=10)
                plt.ylabel("negative log likelihood", labelpad=10)
                # plt.title("Slice along %s (%d samples)" % (xlabel_array[ind], mvec.size), fontsize=24)

            plt.tight_layout()
            plt.savefig(title_plot, bbox_inches='tight')
            plt.show()

        if FLAG_return is True:
            return param_vec, llvec


# Definition of loss functions
# New functions to calculate probabilities correctly
def model_probabilities_np(J_num, uvec, w0, w1, w2, tvec):
    '''
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    '''

    Jix = J_num[0]
    Jiy = J_num[1]
    Jiz = J_num[2]
    Jzx = J_num[3]
    Jzy = J_num[4]
    Jzz = J_num[5]

    # Parameters that depend on J
    a = Jiz + ((-1)**uvec)*Jzz
    abs_beta = np.sqrt( (Jix ** 2 + ((-1) ** uvec) * 2 * Jix * Jzx + Jzx ** 2) +
                        (Jiy ** 2 + ((-1) ** uvec) * 2 * Jiy * Jzy + Jzy ** 2) )

    omega = np.sqrt(a ** 2 + abs_beta ** 2)
    cos_omega_t = np.cos(omega * tvec)
    sin_omega_t = np.sin(omega * tvec)

    delta = np.arcsin(a / omega)
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    den_beta = Jix + ((-1) ** uvec) * Jzx
    num_beta = Jiy + ((-1) ** uvec) * Jzy
    phi = np.arctan2(num_beta, den_beta) # arg(beta)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

    p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

    p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

    fpvec = w0 * p0 + w1 * p1 + w2 * p2
    return fpvec


# New functions to calculate probabilities correctly using param_array
def model_probabilities_param_np(param_num, uvec, w0, w1, w2, tvec):
    '''
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    '''

    omega0 = param_num[0]
    delta0 = param_num[1]
    phi0 = param_num[2]
    omega1 = param_num[3]
    delta1 = param_num[4]
    phi1 = param_num[5]

    # Parameters that depend on J
    omega = (1 - uvec)*omega0 + uvec*omega1
    delta = (1 - uvec)*delta0 + uvec*delta1
    phi = (1 - uvec)*phi0 + uvec*phi1

    cos_omega_t = np.cos(omega * tvec)
    sin_omega_t = np.sin(omega * tvec)

    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

    p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

    p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

    fpvec = w0 * p0 + w1 * p1 + w2 * p2
    return fpvec


def model_probabilities_analytic(tvec, teff=None, J_num=None, param_array=None, FLAG_rabi=True):
    '''
    One function to rule them all!!!

    Calculates probability using J parameters or
    param_array = (omega0, delta0, phi0, omega1, delta1, phi1)
    :param J_num:
    :param tvec:
    :return:
    '''
    if (J_num is None) and (param_array is None):
        print('Requires J_num or param_array as input')

    if teff is None:
        teff = np.array([0, 0])

    if not(J_num is None):
        # Extract from inputs
        Jix = J_num[0]
        Jiy = J_num[1]
        Jiz = J_num[2]
        Jzx = J_num[3]
        Jzy = J_num[4]
        Jzz = J_num[5]

        # Parameters that depend on J
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

    if not(param_array is None):
        omega0, delta0, phi0, omega1, delta1, phi1 = param_array

    # Function to return 2*P(0) - 1
    pvec = np.zeros((6, len(tvec)))

    # Apply phase shifts to the time-stamps
    tvec0 = tvec + teff[0]
    tvec1 = tvec + teff[1]

    # omega0 = np.sqrt((Jiz + Jzz) ** 2 + (Jix + Jzx) ** 2 + (Jiy + Jzy) ** 2)
    # omega1 = np.sqrt((Jiz - Jzz) ** 2 + (Jix - Jzx) ** 2 + (Jiy - Jzy) ** 2)
    # delta0 = np.arcsin((Jiz + Jzz) / omega0)
    # delta1 = np.arcsin((Jiz - Jzz) / omega1)
    # phi0 = np.arctan2((Jiy + Jzy), (Jix + Jzx))
    # phi1 = np.arctan2((Jiy - Jzy), (Jix - Jzx))

    pvec[0, :] = 0.5 * ((np.cos(omega0 * tvec0) + np.sin(phi0) * np.cos(delta0) * np.sin(omega0 * tvec0)) ** 2 + (
                np.sin(delta0) * np.sin(omega0 * tvec0) + np.cos(phi0) * np.cos(delta0) * np.sin(omega0 * tvec0)) ** 2)

    pvec[1, :] = 0.5 * ((np.cos(omega1 * tvec1) + np.sin(phi1) * np.cos(delta1) * np.sin(omega1 * tvec1)) ** 2 + (
            np.sin(delta1) * np.sin(omega1 * tvec1) + np.cos(phi1) * np.cos(delta1) * np.sin(omega1 * tvec1)) ** 2)

    pvec[2, :] = 0.5 * ((np.cos(omega0 * tvec0) - np.cos(phi0) * np.cos(delta0) * np.sin(omega0 * tvec0)) ** 2 + (
                np.sin(delta0) * np.sin(omega0 * tvec0) + np.sin(phi0) * np.cos(delta0) * np.sin(omega0 * tvec0)) ** 2)

    pvec[3, :] = 0.5 * ((np.cos(omega1 * tvec1) - np.cos(phi1) * np.cos(delta1) * np.sin(omega1 * tvec1)) ** 2 + (
            np.sin(delta1) * np.sin(omega1 * tvec1) + np.sin(phi1) * np.cos(delta1) * np.sin(omega1 * tvec1)) ** 2)

    pvec[4, :] = 1.0 - (np.cos(delta0) ** 2) * (np.sin(omega0 * tvec0) ** 2)
    pvec[5, :] = 1.0 - (np.cos(delta1) ** 2) * (np.sin(omega1 * tvec1) ** 2)

    # This is done so as to not break existing code for the moment but this needs to be updated!!!
    if FLAG_rabi is True:
        return 2*pvec - 1
    else:
        return pvec

    # pvec[0, :] = 2 * (np.sin(phi0)*np.cos(omega0 * tvec) +
    #                   np.sin(delta0) * np.cos(phi0) * np.sin(omega0 * tvec))*np.cos(delta0)*np.sin(omega0*tvec)
    #
    # pvec[1, :] = 2 * (np.sin(phi1) * np.cos(omega1 * tvec) +
    #                   np.sin(delta1) * np.cos(phi1) * np.sin(omega1 * tvec)) * np.cos(delta1) * np.sin(omega1 * tvec)
    #
    # pvec[2, :] = -2*(np.cos(phi0)*np.cos(omega0*tvec) -
    #                  np.sin(delta0)*np.sin(phi0)*np.sin(omega0*tvec)) * np.cos(delta0) * np.sin(omega0 * tvec)
    #
    # pvec[3, :] = -2 * (np.cos(phi1) * np.cos(omega1 * tvec) -
    #                    np.sin(delta1) * np.sin(phi1) * np.sin(omega1 * tvec)) * np.cos(delta1) * np.sin(omega1 * tvec)
    #
    # pvec[4, :] = 1.0 - 2*(np.cos(delta0) ** 2) * (np.sin(omega0 * tvec) ** 2)
    # pvec[5, :] = 1.0 - 2*(np.cos(delta1) ** 2) * (np.sin(omega1 * tvec) ** 2)

    # return pvec


def model_probabilities_tf(J_num, uvec, w0, w1, w2, tvec):
    '''
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    '''

    Jix = J_num[0]
    Jiy = J_num[1]
    Jiz = J_num[2]
    Jzx = J_num[3]
    Jzy = J_num[4]
    Jzz = J_num[5]

    # Parameters that depend on J
    a = Jiz + ((-1)**uvec)*Jzz
    abs_beta = tf.sqrt( (Jix ** 2 + ((-1) ** uvec) * 2 * Jix * Jzx + Jzx ** 2) +
                        (Jiy ** 2 + ((-1) ** uvec) * 2 * Jiy * Jzy + Jzy ** 2) )

    omega = tf.sqrt(a ** 2 + abs_beta ** 2)
    cos_omega_t = tf.cos(omega * tvec)
    sin_omega_t = tf.sin(omega * tvec)

    delta = tf.asin(a / omega)
    sin_delta = tf.sin(delta)
    cos_delta = tf.cos(delta)

    den_beta = Jix + ((-1) ** uvec) * Jzx
    num_beta = Jiy + ((-1) ** uvec) * Jzy
    phi = tf.atan2(num_beta, den_beta) # arg(beta)

    cos_phi = tf.cos(phi)
    sin_phi = tf.sin(phi)

    p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

    p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

    p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

    fpvec = w0 * p0 + w1 * p1 + w2 * p2
    return fpvec


def model_probabilities_param_tf(param_num, uvec, w0, w1, w2, tvec_expt):
    '''
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    '''

    omega0 = param_num[0]
    delta0 = param_num[1]
    phi0 = param_num[2]
    omega1 = param_num[3]
    delta1 = param_num[4]
    phi1 = param_num[5]

    # Parameters that depend on J
    omega = (1 - uvec)*omega0 + uvec*omega1
    delta = (1 - uvec)*delta0 + uvec*delta1
    phi = (1 - uvec)*phi0 + uvec*phi1

    # Time information
    coeff = 1.423e-8
    teff0 = 3.1458 / (omega0 + coeff * omega0 ** 2)
    teff1 = 3.1458 / (omega1 + coeff * omega1 ** 2)

    tvec = tvec_expt + (1 - uvec)*teff0 + uvec*teff1

    cos_omega_t = tf.cos(omega * tvec)
    sin_omega_t = tf.sin(omega * tvec)


    sin_delta = tf.sin(delta)
    cos_delta = tf.cos(delta)

    cos_phi = tf.cos(phi)
    sin_phi = tf.sin(phi)

    p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

    p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

    p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

    fpvec = w0 * p0 + w1 * p1 + w2 * p2
    return fpvec


# For old ibm devices that were never online in the first place
def model_probabilities_param_tf_nd(param_num_nd, xi_param, uvec, w0, w1, w2, tvec_expt):
    '''
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    '''
    param_num = param_num_nd*xi_param

    omega0 = param_num[0]
    delta0 = param_num[1]
    phi0 = param_num[2]
    omega1 = param_num[3]
    delta1 = param_num[4]
    phi1 = param_num[5]

    # Parameters that depend on J
    omega = (1 - uvec)*omega0 + uvec*omega1
    delta = (1 - uvec)*delta0 + uvec*delta1
    phi = (1 - uvec)*phi0 + uvec*phi1

    # Time information
    coeff = 1.423e-8
    teff0 = 3.1458 / (omega0 + coeff * omega0 ** 2)
    teff1 = 3.1458 / (omega1 + coeff * omega1 ** 2)

    tvec = tvec_expt + (1 - uvec)*teff0 + uvec*teff1

    cos_omega_t = tf.cos(omega * tvec)
    sin_omega_t = tf.sin(omega * tvec)


    sin_delta = tf.sin(delta)
    cos_delta = tf.cos(delta)

    cos_phi = tf.cos(phi)
    sin_phi = tf.sin(phi)

    p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

    p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

    p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

    fpvec = w0 * p0 + w1 * p1 + w2 * p2
    return fpvec


def model_probabilities_J_tf_nd(J_num_nd, xi_J, uvec, w0, w1, w2, tvec_expt):
    '''
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    '''
    J_num = J_num_nd*xi_J

    Jix = J_num[0]
    Jiy = J_num[1]
    Jiz = J_num[2]
    Jzx = J_num[3]
    Jzy = J_num[4]
    Jzz = J_num[5]

    # Parameters that depend on J
    a = Jiz + ((-1) ** uvec) * Jzz
    abs_beta = tf.sqrt((Jix ** 2 + ((-1) ** uvec) * 2 * Jix * Jzx + Jzx ** 2) +
                       (Jiy ** 2 + ((-1) ** uvec) * 2 * Jiy * Jzy + Jzy ** 2))

    omega = tf.sqrt(a ** 2 + abs_beta ** 2)

    # Time information
    coeff = 1.423e-8
    teff = 3.1458 / (omega + coeff * omega ** 2)
    tvec = tvec_expt + teff

    cos_omega_t = tf.cos(omega * tvec)
    sin_omega_t = tf.sin(omega * tvec)

    delta = tf.asin(a / omega)
    sin_delta = tf.sin(delta)
    cos_delta = tf.cos(delta)

    den_beta = Jix + ((-1) ** uvec) * Jzx
    num_beta = Jiy + ((-1) ** uvec) * Jzy
    phi = tf.atan2(num_beta, den_beta)  # arg(beta)

    cos_phi = tf.cos(phi)
    sin_phi = tf.sin(phi)

    p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

    p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

    p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

    fpvec = w0 * p0 + w1 * p1 + w2 * p2
    return fpvec


def model_probabilities_ibmq_np_nd(J_num_nd, type_param, xi_J, uvec, w0, w1, w2, tvec_expt):
    '''
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    '''
    J_num = J_num_nd*xi_J

    if type_param == 'J':
        Jix = J_num[0]
        Jiy = J_num[1]
        Jiz = J_num[2]
        Jzx = J_num[3]
        Jzy = J_num[4]
        Jzz = J_num[5]

        # Parameters that depend on J
        a = Jiz + ((-1) ** uvec) * Jzz
        abs_beta = np.sqrt((Jix ** 2 + ((-1) ** uvec) * 2 * Jix * Jzx + Jzx ** 2) +
                           (Jiy ** 2 + ((-1) ** uvec) * 2 * Jiy * Jzy + Jzy ** 2))

        omega = np.sqrt(a ** 2 + abs_beta ** 2)

        delta = np.arcsin(a / omega)

        den_beta = Jix + ((-1) ** uvec) * Jzx
        num_beta = Jiy + ((-1) ** uvec) * Jzy
        phi = np.arctan2(num_beta, den_beta)  # arg(beta)

    elif type_param == 'Theta':
        omega0 = J_num[0]
        delta0 = J_num[1]
        phi0 = J_num[2]
        omega1 = J_num[3]
        delta1 = J_num[4]
        phi1 = J_num[5]

        # Parameters that depend on J
        omega = (1 - uvec) * omega0 + uvec * omega1
        delta = (1 - uvec) * delta0 + uvec * delta1
        phi = (1 - uvec) * phi0 + uvec * phi1
    else:
        raise RuntimeError('Wrong Parameterization input!')

    # Time information
    coeff = 1.50856579e-09
    teff = 6.27739558 / (omega + coeff * omega ** 2)
    tvec = tvec_expt + teff

    cos_omega_t = np.cos(omega * tvec)
    sin_omega_t = np.sin(omega * tvec)


    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

    p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

    p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

    fpvec = w0 * p0 + w1 * p1 + w2 * p2

    return fpvec


# For new ibm device ibmq_boel
def model_probabilities_J_ibmq_tf_nd(J_num_nd, xi_J, uvec, w0, w1, w2, tvec_expt):
    '''
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    '''
    J_num = J_num_nd*xi_J

    Jix = J_num[0]
    Jiy = J_num[1]
    Jiz = J_num[2]
    Jzx = J_num[3]
    Jzy = J_num[4]
    Jzz = J_num[5]

    # Parameters that depend on J
    a = Jiz + ((-1) ** uvec) * Jzz
    abs_beta = tf.sqrt((Jix ** 2 + ((-1) ** uvec) * 2 * Jix * Jzx + Jzx ** 2) +
                       (Jiy ** 2 + ((-1) ** uvec) * 2 * Jiy * Jzy + Jzy ** 2))

    # Added a small constant so derivative of sqrt doesn't become infinity when Jiz \approx Jzz and uvec = 1
    omega = tf.sqrt(a ** 2 + abs_beta ** 2)

    # Time information
    coeff = 1.50856579e-09
    teff = 6.27739558 / (omega + coeff * omega ** 2)
    tvec = tvec_expt + teff

    cos_omega_t = tf.cos(omega * tvec)
    sin_omega_t = tf.sin(omega * tvec)

    delta = tf.asin(a / omega)
    sin_delta = tf.sin(delta)
    cos_delta = tf.cos(delta)

    den_beta = Jix + ((-1) ** uvec) * Jzx
    num_beta = Jiy + ((-1) ** uvec) * Jzy
    phi = tf.atan2(num_beta, den_beta)  # arg(beta)

    cos_phi = tf.cos(phi)
    sin_phi = tf.sin(phi)

    p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

    p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

    p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

    fpvec = w0 * p0 + w1 * p1 + w2 * p2

    return fpvec


def model_probabilities_param_ibmq_tf_nd(param_num_nd, xi_param, uvec, w0, w1, w2, tvec_expt):
    '''
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    '''
    param_num = param_num_nd*xi_param

    omega0 = param_num[0]
    delta0 = param_num[1]
    phi0 = param_num[2]
    omega1 = param_num[3]
    delta1 = param_num[4]
    phi1 = param_num[5]

    # Parameters that depend on J
    omega = (1 - uvec)*omega0 + uvec*omega1
    delta = (1 - uvec)*delta0 + uvec*delta1
    phi = (1 - uvec)*phi0 + uvec*phi1

    # Time information
    coeff = 1.50856579e-09
    teff0 = 6.27739558 / (omega0 + coeff * omega0 ** 2)
    teff1 = 6.27739558 / (omega1 + coeff * omega1 ** 2)

    tvec = tvec_expt + (1 - uvec)*teff0 + uvec*teff1

    cos_omega_t = tf.cos(omega * tvec)
    sin_omega_t = tf.sin(omega * tvec)


    sin_delta = tf.sin(delta)
    cos_delta = tf.cos(delta)

    cos_phi = tf.cos(phi)
    sin_phi = tf.sin(phi)

    p0 = 0.5 * ((cos_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + cos_phi * cos_delta * sin_omega_t) ** 2)

    p1 = 0.5 * ((cos_omega_t - cos_phi * cos_delta * sin_omega_t) ** 2 +
                (sin_delta * sin_omega_t + sin_phi * cos_delta * sin_omega_t) ** 2)

    p2 = cos_omega_t ** 2 + (sin_delta * sin_omega_t) ** 2

    fpvec = w0 * p0 + w1 * p1 + w2 * p2

    return fpvec


# Shift to learner_experiments.py later!
def mle_estimation_passive_learner_expt_data(env_cr, moset, prepset, time_stamps, freq_convert, xi_t, xi_J,
                                        N_0=972, N_batch=486, max_iter = 10, FLAG_initial_estimate = True,
                                             verbose=True, do_plot=False, log_file=None):
    '''
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
    '''

    # Create some data structures
    time_stamps_nd = time_stamps / xi_t

    # Initialize action space
    A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps_nd, xi_t, xi_J=xi_J, freq_convert=freq_convert)

    # Should create an environment (simulated) using the true value of J
    # to pass to sample_action_space for querying and creating the dataset
    J_truth = np.array([env_cr.Jix, env_cr.Jiy, env_cr.Jiz, env_cr.Jzx, env_cr.Jzy, env_cr.Jzz])

    # Variable definitions
    N_config = A_cr.N_actions  # Number of different configurations/actions in the pool
    loss_num = []
    J_num = []
    J_nd_num = []
    N_p_vec = []
    q_vec = []

    # Uniform probability distribution over the pool
    p_U = (1/N_config)*np.ones(N_config)

    n_samples_query_U = round(N_0 / N_config, 1)
    set_U = n_samples_query_U * np.ones(N_config)

    # Create initial dataset using set_U (and not p_U -- generalize later)
    X_p = A_cr.sample_action_space(env_cr, set_U, N_0, FLAG_query=False)
    N_p = N_0

    # Update action space with actions sampled
    A_cr.update_dict_action_space(X_p)

    # Get estimate of J
    mini_batch_size = np.amin([int(N_p/5), 512])
    solver_options = {'mini_batch_size': mini_batch_size}
    _loss, _Jnum = estimation_procedure(X_p, env_cr=env_cr, A_cr=A_cr, solver_options=solver_options,
                                        verbose=verbose, do_plot=do_plot, FLAG_syn_data=False,
                                        FLAG_initial_estimate=FLAG_initial_estimate)

    loss_num.append(_loss)
    J_num.append(_Jnum)
    J_nd_num.append(_Jnum/xi_J)
    N_p_vec.append(N_p)
    q_vec.append(p_U)

    # Write to log file
    if log_file is not None:
        f_log = open(log_file, "a+")
        mse_U_temp = normalized_L2_error(_Jnum, J_truth, xi_J) ** 2
        f_log.write("%d %f %f \n" % (0, np.sqrt(mse_U_temp), _loss))
        f_log.close()

    print('Uniform query distribution')

    # Online optimization
    for k in range(max_iter):
        # Uniform probability distribution over the pool
        n_samples_query_U = round(N_batch / N_config, 1)
        set_U = n_samples_query_U * np.ones(N_config)

        # Sample from query distribution
        X_q = A_cr.sample_action_space(env_cr, set_U, N_batch, FLAG_query=False)

        # Update action space with actions sampled
        A_cr.update_dict_action_space(X_q)

        # Update data with queried dataset -- merge_datasets will probably need to be updated when we have adaptive time_stamps
        X_p = A_cr.merge_datasets(X_p, X_q)
        N_p = N_p + N_batch

        # Update MLE
        #init_J_running = J_nd_num[n]

        solver_options['mini_batch_size'] = np.amin([int(N_p / 5), 512])
        _loss, _Jnum = estimation_procedure(X_p, init_J=J_num[k], env_cr=env_cr,  A_cr=A_cr, solver_options=solver_options,
                                            verbose=verbose, do_plot=do_plot, FLAG_syn_data=False,
                                        FLAG_initial_estimate=FLAG_initial_estimate)

        loss_num.append(_loss)
        J_num.append(_Jnum)
        J_nd_num.append(_Jnum / xi_J)
        N_p_vec.append(N_p)

        # Write to log file
        if log_file is not None:
            f_log = open(log_file, "a+")
            mse_U_temp = normalized_L2_error(_Jnum, J_truth, xi_J) ** 2
            f_log.write("%d %f %f \n" % (k + 1, np.sqrt(mse_U_temp), _loss))
            f_log.close()

    # Note results return non-normalized variables
    results = {'loss': loss_num, 'J_hat': J_num,
               'J_truth': np.array([env_cr.Jix, env_cr.Jiy, env_cr.Jiz, env_cr.Jzx, env_cr.Jzy, env_cr.Jzz]),
               'N_p': N_p_vec, 'data': X_p, 'xi_J': xi_J, 'A_cr': A_cr}

    return results


def mle_estimation_active_learner_expt_data(env_cr, moset, prepset, time_stamps, freq_convert, xi_t, xi_J,
                                            active_learner, FLAG_initial_estimate = True,
                                            FLAG_constraints=False, query_constraints_ref=None,
                                            FLAG_lower_limits = False,
                                            growth_time_stamps=None, max_iter_growth=10, growth_factor=2,
                                            N_0=972, N_batch=486, max_iter = 10, FLAG_debug_AL=False,
                                            verbose=True, do_plot=False, log_file=None):
    '''
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
    '''

    # Create some data structures
    time_stamps_nd = time_stamps / xi_t

    # Initialize action space
    A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps_nd, xi_t, xi_J=xi_J, freq_convert=freq_convert)

    # Should create an environment (simulated) using the true value of J
    # to pass to sample_action_space for querying and creating the dataset
    J_truth = np.array([env_cr.Jix, env_cr.Jiy, env_cr.Jiz, env_cr.Jzx, env_cr.Jzy, env_cr.Jzz])

    # Variable definitions
    N_config = A_cr.N_actions  # Number of different configurations/actions in the pool
    loss_num = []
    J_num = []
    J_nd_num = []
    N_p_vec = []
    q_vec = []

    # Uniform probability distribution over the pool
    p_U = (1/N_config)*np.ones(N_config)

    n_samples_query_U = round(N_0 / N_config)
    set_P = n_samples_query_U * np.ones(N_config)   # Number of shots made so far for each query
    set_Q = np.zeros(N_config)  # Number of shots being considered for each query

    # Mixing parameter for active learning procedure (See Chaudhuri et al., 2015; Sourati et al, 2017)
    lambda_m = 1.0 - 1. / ((N_batch) ** (1 / 6))

    # Create initial dataset using set_P (and not p_U -- generalize later)
    X_p = A_cr.sample_action_space(env_cr, set_P, N_0, FLAG_query=False)

    N_p = N_0

    # Update action space with actions sampled
    A_cr.update_dict_action_space(X_p)

    # Get estimate of J
    if FLAG_debug_AL:
        init_J_num = J_truth
        FLAG_initial_estimate = False
    else:
        init_J_num = None
        FLAG_initial_estimate = True

    mini_batch_size = np.amin([int(N_p/5), 512])
    solver_options = {'mini_batch_size': mini_batch_size}
    _loss, _Jnum = estimation_procedure(X_p, init_J=init_J_num, env_cr=env_cr, A_cr=A_cr, solver_options=solver_options,
                                        verbose=verbose, do_plot=do_plot, FLAG_syn_data=False,
                                        FLAG_initial_estimate=FLAG_initial_estimate)

    loss_num.append(_loss)
    J_num.append(_Jnum)
    J_nd_num.append(_Jnum/xi_J)
    N_p_vec.append(N_p)
    q_vec.append(p_U)

    # Write to log file
    if log_file is not None:
        f_log = open(log_file, "a+")
        mse_U_temp = normalized_L2_error(_Jnum, J_truth, xi_J) ** 2
        f_log.write("%d %f %f \n" % (0, np.sqrt(mse_U_temp), _loss))
        f_log.close()

    # Create Quantum System Model based on the estimate so far
    # Noise Model
    qs_noise = {'readout': env_cr.readout_noise,
                'control_noise': functools.partial(quantum_device_models.data_driven_teff_noise_model,
                                                   FLAG_ibmq_boel=True),
                'grad_control_noise': functools.partial(quantum_device_models.grad_data_driven_teff_noise_model,
                                                        FLAG_ibmq_boel=True)}

    qs_model = quantum_device_models.SystemModel(_Jnum, 1e6 * np.ones(len(_Jnum)), noise=qs_noise,
                                                       FLAG_readout_noise=True, FLAG_control_noise=True)

    print('Optimal query distribution')

    # For adaptive query space
    DT = np.abs(time_stamps[-1] - time_stamps[0])
    dt = np.mean(time_stamps[1:] - time_stamps[0:-1]) # if time step size isn't uniform in the initial range

    n_iter_growth = 0
    k_growth = 1

    # ActiveLearning + Online optimization
    for k in range(max_iter):
        # Update counter to check if it's time to grow the query space or not
        n_iter_growth += 1

        if growth_time_stamps is not None and n_iter_growth > max_iter_growth:
            k_growth += 1
            # Update the time_stamps
            # Note that the resolution in the old range may not necessarily be the same in the new range
            # for both the simulator oracle and the experimental data oracle
            print('Updating query space!')

            # Concatenatation ensures old tvec doesn't change and the A_cr is able to work with old training examples
            if growth_time_stamps == 'linear':
                time_stamps_mod = np.arange(time_stamps[-1]+dt, time_stamps[0] + growth_factor * DT, dt)

            elif growth_time_stamps == 'exponential':
                time_stamps_mod = np.arange(time_stamps[-1]+dt, time_stamps[0] + (growth_factor ** (k - 1)) * DT, dt)
            else:
                raise RuntimeError('Passed invalid growth argument error')

            time_stamps = np.concatenate((time_stamps, time_stamps_mod), axis=0)
            time_stamps_nd = time_stamps / xi_t

            # Update the action space
            A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps_nd, xi_t,
                                                xi_J=qs_model.xi_J, xi_param=qs_model.xi_param,
                                                freq_convert=freq_convert)

            N_config = A_cr.N_actions

            # Update the dictionary of action space with the samples we already have
            A_cr.update_dict_action_space(X_p)

            # Update uniform query distribution as support has now changed
            p_U = (1 / N_config) * np.ones(N_config)

            # Reset the counter for growth
            n_iter_growth = 0

            # Switch off FLAG_initial_estimate
            FLAG_initial_estimate = False

        # Update the ActiveLearner -- Query Constraints and (soon: adaptively growing query space)
        if FLAG_constraints:
            N_tot = N_p + N_batch
            if FLAG_lower_limits:
                query_constraints = {'N_tot_old': N_p, 'N_tot': N_tot,
                                     'q_old': q_vec[-1], 'N_shots': query_constraints_ref['N_shots'], 'FLAG_lower_limits': True}
            else:
                upper_bound_q = [A_cr.action_n_shots[ind]/A_cr.max_n_shots for ind in range(A_cr.N_actions)]
                query_constraints = {'N_tot_old': N_p, 'N_tot': N_tot,
                                     'q_old': q_vec[-1], 'upper_bound_q': upper_bound_q, 'FLAG_lower_limits': False}

            active_learner.update(FLAG_constraints=FLAG_constraints, query_constraints=query_constraints)

        # Query optimization
        q = active_learner.optimal_query_distribution(time_stamps_nd, xi_t, N_config, qs_model, p_ref=p_U)
        q = lambda_m * q + (1 - lambda_m) * p_U

        # Sample from query distribution q noting that some queries have already been made
        X_q = A_cr.sample_action_space(env_cr, q, N_batch, FLAG_query=True)

        # Update action space with actions sampled
        A_cr.update_dict_action_space(X_q)

        # Update data with queried dataset -- merge_datasets will probably need to be updated when we have adaptive time_stamps
        X_p = A_cr.merge_datasets(X_p, X_q)
        N_p = N_p + N_batch

        # Update MLE
        solver_options['mini_batch_size'] = np.amin([int(N_p / 5), 512])

        if FLAG_debug_AL:
            init_J_num = J_truth
            FLAG_initial_estimate = False
        else:
            init_J_num = J_num[k]

        _loss, _Jnum = estimation_procedure(X_p, init_J=init_J_num, env_cr=env_cr, A_cr=A_cr,
                                            solver_options=solver_options,
                                            verbose=verbose, do_plot=do_plot, FLAG_syn_data=False,
                                            FLAG_initial_estimate=FLAG_initial_estimate)

        # Update SystemModel
        qs_model.update(_Jnum)

        # Update variables
        loss_num.append(_loss)
        J_num.append(_Jnum)
        J_nd_num.append(_Jnum / xi_J)
        N_p_vec.append(N_p)
        q_vec.append(q)

        # Write to log file
        if log_file is not None:
            f_log = open(log_file, "a+")
            mse_U_temp = normalized_L2_error(_Jnum, J_truth, xi_J) ** 2
            f_log.write("%d %f %f \n" % (k+1, np.sqrt(mse_U_temp), _loss))
            f_log.close()

    results = {'loss': loss_num, 'J_hat': J_num,
               'J_truth': np.array([env_cr.Jix, env_cr.Jiy, env_cr.Jiz, env_cr.Jzx, env_cr.Jzy, env_cr.Jzz]),
               'N_p': N_p_vec, 'q': q_vec, 'data': X_p, 'xi_J': xi_J, 'A_cr': A_cr}

    return results


def mle_estimation_single_step_query_solve(env_cr, moset, prepset, time_stamps, freq_convert, xi_t, xi_J,
                                            active_learner, FLAG_initial_estimate = True,
                                            FLAG_constraints=False, query_constraints_ref=None,
                                            FLAG_lower_limits = False,
                                            growth_time_stamps=None, max_iter_growth=10, growth_factor=2,
                                            N_0=972, N_batch=486, max_iter = 10, FLAG_debug_AL=False,
                                            verbose=True, do_plot=False, log_file=None):
    '''
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
    '''

    # Create some data structures
    time_stamps_nd = time_stamps / xi_t

    # Initialize action space
    A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps_nd, xi_t, xi_J=xi_J, freq_convert=freq_convert)

    # Should create an environment (simulated) using the true value of J
    # to pass to sample_action_space for querying and creating the dataset
    J_truth = np.array([env_cr.Jix, env_cr.Jiy, env_cr.Jiz, env_cr.Jzx, env_cr.Jzy, env_cr.Jzz])

    # Variable definitions
    N_config = A_cr.N_actions  # Number of different configurations/actions in the pool
    loss_num = []
    J_num = []
    J_nd_num = []
    N_p_vec = []
    q_vec = []

    # Uniform probability distribution over the pool
    p_U = (1 / N_config) * np.ones(N_config)

    n_samples_query_U = round(N_0 / N_config)
    set_P = n_samples_query_U * np.ones(N_config)  # Number of shots made so far for each query
    set_Q = np.zeros(N_config)  # Number of shots being considered for each query

    # Mixing parameter for active learning procedure (See Chaudhuri et al., 2015; Sourati et al, 2017)
    lambda_m = 1.0 - 1. / ((N_batch) ** (1 / 6))

    # Create initial dataset using set_P (and not p_U -- generalize later)
    X_p = A_cr.sample_action_space(env_cr, set_P, N_0, FLAG_query=False)

    N_p = N_0

    # Update action space with actions sampled
    A_cr.update_dict_action_space(X_p)

    # Get estimate of J
    if FLAG_debug_AL:
        init_J_num = J_truth
        FLAG_initial_estimate = False
    else:
        init_J_num = None
        FLAG_initial_estimate = True

    mini_batch_size = np.amin([int(N_p / 5), 512])
    solver_options = {'mini_batch_size': mini_batch_size}
    _loss, _Jnum = estimation_procedure(X_p, init_J=J_truth, env_cr=env_cr, A_cr=A_cr, solver_options=solver_options,
                                        verbose=verbose, do_plot=do_plot, FLAG_syn_data=False,
                                        FLAG_initial_estimate=FLAG_initial_estimate)

    loss_num.append(_loss)
    J_num.append(_Jnum)
    J_nd_num.append(_Jnum/xi_J)
    N_p_vec.append(N_p)
    q_vec.append(p_U)

    # Create Quantum System Model based on the estimate so far
    # Noise Model
    qs_noise = {'readout': env_cr.readout_noise,
                'control_noise': functools.partial(quantum_device_models.data_driven_teff_noise_model,
                                                   FLAG_ibmq_boel=True),
                'grad_control_noise': functools.partial(quantum_device_models.grad_data_driven_teff_noise_model,
                                                        FLAG_ibmq_boel=True)}

    qs_model = quantum_device_models.SystemModel(_Jnum, 1e6 * np.ones(len(_Jnum)), noise=qs_noise,
                                                       FLAG_readout_noise=True, FLAG_control_noise=True)

    print('Optimal query distribution')

    # For adaptive query space
    DT = np.abs(time_stamps[-1] - time_stamps[0])
    dt = np.mean(time_stamps[1:] - time_stamps[0:-1]) # if time step size isn't uniform in the initial range

    # One step of ActiveLearning + Online optimization
    if FLAG_constraints:
        N_tot = N_p + N_batch
        if FLAG_lower_limits:
            query_constraints = {'N_tot_old': N_p, 'N_tot': N_tot,
                                 'q_old': q_vec[-1], 'N_shots': query_constraints_ref['N_shots'], 'FLAG_lower_limits': True}
        else:
            upper_bound_q = [A_cr.action_n_shots[ind]/A_cr.max_n_shots for ind in range(A_cr.N_actions)]
            query_constraints = {'N_tot_old': N_p, 'N_tot': N_tot,
                                 'q_old': q_vec[-1], 'upper_bound_q': upper_bound_q, 'FLAG_lower_limits': False}

        active_learner.update(FLAG_constraints=FLAG_constraints, query_constraints=query_constraints)

    # Query optimization
    q = active_learner.optimal_query_distribution(time_stamps_nd, xi_t, N_config, qs_model, p_ref=p_U)

    # Update variables
    loss_num.append(_loss)
    J_num.append(_Jnum)
    J_nd_num.append(_Jnum / xi_J)
    N_p_vec.append(N_p)
    q_vec.append(q)

    results = {'loss': loss_num, 'J_hat': J_num,
               'J_truth': np.array([env_cr.Jix, env_cr.Jiy, env_cr.Jiz, env_cr.Jzx, env_cr.Jzy, env_cr.Jzz]),
               'N_p': N_p_vec, 'q': q_vec, 'data': X_p, 'xi_J': xi_J, 'A_cr': A_cr}

    return results


def random_policy_fisher_information(i_run, qs_model, moset, prepset, time_stamps, freq_convert, xi_t, xi_J,
                                     n_batches, max_k, env_qs=None,
                                     growth_time_stamps=None, max_iter_growth=5, growth_factor=2,
                                     N_0=972, N_batch=100, max_iter = 10, FLAG_debug_AL=False,
                                     verbose=True, do_plot=False, log_file=None):
    '''
    Computes the Cramer-Rao bound for a random policy

    Functionality required:
    - compute Fisher information for a given list of actions (M,U,t)
    - given a list of actions, filter and return those with the maximum entropy
    - given a list of actions and value of param-array, filter and return those which are close to the rabi oscillation zero-crossings
    - Do the above lists of actions overlap?
    - sample from a list of actions according to some distribution (uniform to begin with)
    '''

    np.random.seed(10 * (i_run + 2))
    # Create some data structures
    time_stamps_nd = time_stamps / xi_t

    RMSE_param_array = np.zeros(max_k)

    # Running Fisher Information matrix
    FI_param_running = np.zeros(shape=(6, 6))

    # Initialize action space
    # A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps, xi_t, xi_param)
    A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps, xi_t,
                                        xi_J=qs_model.xi_J, xi_param=qs_model.xi_param, freq_convert=freq_convert)

    # Variable definitions
    N_config = A_cr.N_actions  # Number of different configurations/actions in the pool
    loss_num = []
    J_num = []
    J_nd_num = []
    N_p_vec = []
    q_vec = []

    # Uniform probability distribution over the pool
    p_U = (1/N_config)*np.ones(N_config)

    n_samples_query_U = round(N_0 / N_config)
    set_P = n_samples_query_U * np.ones(N_config)   # Number of shots made so far for each query

    # Mixing parameter for active learning procedure (See Chaudhuri et al., 2015; Sourati et al, 2017)
    lambda_m = 1.0 - 1. / ((N_batch) ** (1 / 6))

    # Create initial dataset using set_P (and not p_U -- generalize later)
    X_p = A_cr.sample_action_space(env_cr, set_P, N_0, FLAG_query=False)

    N_p = N_0

    # Update action space with actions sampled
    A_cr.update_dict_action_space(X_p)

    # Get estimate of J
    if FLAG_debug_AL:
        init_J_num = J_truth
        FLAG_initial_estimate = False
    else:
        init_J_num = None
        FLAG_initial_estimate = True

    mini_batch_size = np.amin([int(N_p/5), 512])
    solver_options = {'mini_batch_size': mini_batch_size}
    _loss, _Jnum = estimation_procedure(X_p, init_J=init_J_num, env_cr=env_cr, A_cr=A_cr, solver_options=solver_options,
                                        verbose=verbose, do_plot=do_plot, FLAG_syn_data=False,
                                        FLAG_initial_estimate=FLAG_initial_estimate)

    loss_num.append(_loss)
    J_num.append(_Jnum)
    J_nd_num.append(_Jnum/xi_J)
    N_p_vec.append(N_p)
    q_vec.append(p_U)

    # Write to log file
    if log_file is not None:
        f_log = open(log_file, "a+")
        mse_U_temp = normalized_L2_error(_Jnum, J_truth, xi_J) ** 2
        f_log.write("%d %f %f \n" % (0, np.sqrt(mse_U_temp), _loss))
        f_log.close()

    # Create Quantum System Model based on the estimate so far
    # Noise Model
    qs_noise = {'readout': env_cr.readout_noise,
                'control_noise': functools.partial(quantum_device_models.data_driven_teff_noise_model,
                                                   FLAG_ibmq_boel=True),
                'grad_control_noise': functools.partial(quantum_device_models.grad_data_driven_teff_noise_model,
                                                        FLAG_ibmq_boel=True)}

    qs_model = quantum_device_models.SystemModel(_Jnum, 1e6 * np.ones(len(_Jnum)), noise=qs_noise,
                                                       FLAG_readout_noise=True, FLAG_control_noise=True)

    print('Optimal query distribution')

    # For adaptive query space
    DT = np.abs(time_stamps[-1] - time_stamps[0])
    dt = np.mean(time_stamps[1:] - time_stamps[0:-1]) # if time step size isn't uniform in the initial range

    n_iter_growth = 0

    # ActiveLearning + Online optimization
    for k in range(max_iter):
        # Update counter to check if it's time to grow the query space or not
        n_iter_growth += 1

        if growth_time_stamps is not None and n_iter_growth > max_iter_growth:
            # Update the time_stamps
            # Note that the resolution in the old range may not necessarily be the same in the new range
            # for both the simulator oracle and the experimental data oracle
            print('Updating query space!')
            # Concatenatation ensures old tvec doesn't change and the A_cr is able to work with old training examples
            if growth_time_stamps == 'linear':
                time_stamps_mod = np.arange(time_stamps[-1]+dt, time_stamps[0] + growth_factor * DT, dt)

            elif growth_time_stamps == 'exponential':
                time_stamps_mod = np.arange(time_stamps[-1]+dt, time_stamps[0] + (growth_factor ** (k - 1)) * DT, dt)
            else:
                raise RuntimeError('Passed invalid growth argument error')

            time_stamps = np.concatenate((time_stamps, time_stamps_mod), axis=0)
            time_stamps_nd = time_stamps / xi_t

            # Update the action space
            A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps_nd, xi_t,
                                                xi_J=qs_model.xi_J, xi_param=qs_model.xi_param,
                                                freq_convert=freq_convert)

            N_config = A_cr.N_actions

            # Update the dictionary of action space with the samples we already have
            A_cr.update_dict_action_space(X_p)

            # Update uniform query distribution as support has now changed
            p_U = (1 / N_config) * np.ones(N_config)

            # Reset the counter for growth
            n_iter_growth = 0

            # Switch off FLAG_initial_estimate
            FLAG_initial_estimate = False

        # Update the ActiveLearner -- Query Constraints and (soon: adaptively growing query space)
        if FLAG_constraints:
            N_tot = N_p + N_batch
            if FLAG_lower_limits:
                query_constraints = {'N_tot_old': N_p, 'N_tot': N_tot,
                                     'q_old': q_vec[-1], 'N_shots': query_constraints_ref['N_shots'], 'FLAG_lower_limits': True}
            else:
                upper_bound_q = [A_cr.action_n_shots[ind]/A_cr.max_n_shots for ind in range(A_cr.N_actions)]
                query_constraints = {'N_tot_old': N_p, 'N_tot': N_tot,
                                     'q_old': q_vec[-1], 'upper_bound_q': upper_bound_q, 'FLAG_lower_limits': False}

            active_learner.update(FLAG_constraints=FLAG_constraints, query_constraints=query_constraints)

        # Query optimization
        q = active_learner.optimal_query_distribution(time_stamps_nd, xi_t, N_config, qs_model, p_ref=p_U)
        q = lambda_m * q + (1 - lambda_m) * p_U

        # Sample from query distribution q noting that some queries have already been made
        X_q = A_cr.sample_action_space(env_cr, q, N_batch, FLAG_query=True)

        # Update action space with actions sampled
        A_cr.update_dict_action_space(X_q)

        # Update data with queried dataset -- merge_datasets will probably need to be updated when we have adaptive time_stamps
        X_p = A_cr.merge_datasets(X_p, X_q)
        N_p = N_p + N_batch

        # Update MLE
        solver_options['mini_batch_size'] = np.amin([int(N_p / 5), 512])

        if FLAG_debug_AL:
            init_J_num = J_truth
            FLAG_initial_estimate = False
        else:
            init_J_num = J_num[k]

        _loss, _Jnum = estimation_procedure(X_p, init_J=init_J_num, env_cr=env_cr, A_cr=A_cr,
                                            solver_options=solver_options,
                                            verbose=verbose, do_plot=do_plot, FLAG_syn_data=False,
                                            FLAG_initial_estimate=FLAG_initial_estimate)

        # Update SystemModel
        qs_model.update(_Jnum)

        # Update variables
        loss_num.append(_loss)
        J_num.append(_Jnum)
        J_nd_num.append(_Jnum / xi_J)
        N_p_vec.append(N_p)
        q_vec.append(q)

        # Write to log file
        if log_file is not None:
            f_log = open(log_file, "a+")
            mse_U_temp = normalized_L2_error(_Jnum, J_truth, xi_J) ** 2
            f_log.write("%d %f %f \n" % (k+1, np.sqrt(mse_U_temp), _loss))
            f_log.close()

    results = {'loss': loss_num, 'J_hat': J_num,
               'J_truth': np.array([env_cr.Jix, env_cr.Jiy, env_cr.Jiz, env_cr.Jzx, env_cr.Jzy, env_cr.Jzz]),
               'N_p': N_p_vec, 'q': q_vec, 'data': X_p, 'xi_J': xi_J, 'A_cr': A_cr}

    return results
