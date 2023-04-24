"""
Contains SGD methods to solve the MLE of Hamiltonian Learning using TensorFlow2.0
Preferred for use over mle_estimators_tf1
"""
import numpy as np
import math
import cmath
import scipy.optimize
import scipy.fftpack
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar, curve_fit, leastsq

import tensorflow as tf
tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

import matplotlib.pyplot as plt
import functools

from ..quantum_system_oracles import simulate_nature
from ..quantum_system_models import quantum_device_models

# # Set number of threads of tensorflow
# tf_num_threads=1
# tf.config.threading.set_inter_op_parallelism_threads(tf_num_threads)


# MLE Estimator
class MLE_Estimator(object):
    """
    Sets up the MLE Estimator with different solver options and loss functions

    TODO: Add options for different parameterizations, loss functions, and generalize for adaptively growing qs e.g. misclassif error
    TODO: Fetch noise models from a QuantumSystemModel rather than the dataset
    """

    def __init__(self, data, xi_param, solver_options=None):
        """
        A general setup that doesn't assume we use tensorflow for SGD and hence additional methods can
        be added without modifying this initialization
        """
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

        # Decide on decoherence noise model to be included -- done in a HACKY way
        # See process_data.create_queryable_dataset_ibmq
        # This option should ideally be fetched from a QuantumSystemModel that is connected to the estimator
        if 'FLAG_decoherence' in data.keys():
            self.FLAG_decoherence = data['FLAG_decoherence']
        else:
            self.FLAG_decoherence = False

    def np_loss(self, param_num, type_param='J', FLAG_log_sum_exp=False, verbose=False):
        """
        param_num - array of Hamiltonian parameters
        type_param - parameterization being used
        FLAG_log_sum_exp -- just a FLAG to use the log-sum-exp trick or not (might remove later)
        """
        # model variables
        param_nd = param_num/self.xi_param

        # model to predict sample probability (prob(k=0))
        if self.FLAG_boel:
            if verbose:
                print('Using ibmq_boel device data-driven teff model for %s' % type_param)

            # probability of target measurement being 0
            if self.FLAG_decoherence:
                Y_sp = model_probabilities_decoherence_ibmq_np_nd(param_nd, type_param,
                                                                  self.xi_param, self.uvec,
                                                                  self.w0_vec, self.w1_vec,
                                                                  self.w2_vec, self.tvec)
            else:
                Y_sp = model_probabilities_ibmq_np_nd(param_nd, type_param,
                                                      self.xi_param, self.uvec,
                                                      self.w0_vec, self.w1_vec,
                                                      self.w2_vec, self.tvec)

        # loss function (binary cross-entropy)
        if self.FLAG_classification:
            Y = self.samples

            if not FLAG_log_sum_exp:
                loss = (1 - Y) * ((1 - self.r0) * Y_sp + self.r1 * (1 - Y_sp)) + \
                       Y * ((1 - self.r1) * (1 - Y_sp) + self.r0 * Y_sp)  # likelihood

                loss = -1 * np.log(loss + 1.0e-10) # log likelihood
                loss = np.mean(loss)  # take mean over batch
            else:
                exp_log_Y = np.exp(np.log(Y))
                # max_exp_log_Y = np.amax(exp_log_Y)
                loss = (1 - exp_log_Y) * ((1 - self.r0) * Y_sp + self.r1 * (1 - Y_sp)) + \
                       exp_log_Y * ((1 - self.r1) * (1 - Y_sp) + self.r0 * Y_sp)  # likelihood
                loss = -1 * np.log(loss)
                loss = np.mean(loss)  # take mean over batch
        else:
            loss = self.p_c_given_meas_0*Y_sp + self.p_c_given_meas_1*(1-Y_sp)
            loss = -1 * np.log(loss + 1e-10)
            loss = np.mean(loss)

        return loss

    def log_likelihood_loss_nd(self, param_num_nd, type_param='J', verbose=False):
        """
        Similar to the function above but this loss function used for minimizing using standard out of the box
        estimation strategies such as scipy's optimization routines

        Inputs:
            param_num_nd:
            type_param:
            verbose:
        """
        # probability of target measurement being 0
        # probability of target measurement being 0
        if self.FLAG_decoherence:
            Y_sp = model_probabilities_decoherence_ibmq_np_nd(param_num_nd, type_param,
                                                              self.xi_param, self.uvec,
                                                              self.w0_vec, self.w1_vec,
                                                              self.w2_vec, self.tvec)
        else:
            Y_sp = model_probabilities_ibmq_np_nd(param_num_nd, type_param,
                                                  self.xi_param, self.uvec,
                                                  self.w0_vec, self.w1_vec,
                                                  self.w2_vec, self.tvec)

        if self.FLAG_classification:
            Y = self.samples

            loss = (1 - Y) * ((1 - self.r0) * Y_sp + self.r1 * (1 - Y_sp)) + \
                   Y * ((1 - self.r1) * (1 - Y_sp) + self.r0 * Y_sp)  # likelihood

            loss = -1 * np.log(loss + 1.0e-15)  # log likelihood
            loss = np.mean(loss)  # take mean over batch
        else:
            loss = self.p_c_given_meas_0*Y_sp + self.p_c_given_meas_1*(1-Y_sp)
            loss = -1 * np.log(loss + 1e-15)
            loss = np.mean(loss)

        return loss

    def quasi_newton_solve(self, param_num_nd, FLAG_track_cost=False):
        """
        Solve the MLE with the Quasi-Newton method of L-FBGS-B

        Inputs:
            param_num_nd (Required): initial condition to the solve
        """
        res = scipy.optimize.minimize(self.log_likelihood_loss_nd, param_num_nd,
                                      method='L-BFGS-B', jac=None, bounds=None)

        if FLAG_track_cost:
            cost = res.nfev + 2*res.njev
            return res.x, cost
        else:
            return res.x

    def analytic_loss_gradients(self, param_num, qs_model, train_inputs=None, type_param='Theta', verbose=True):
        """
        Inputs:
        qs_model - Unlike the other functions here, this requires passing in the quantum system model so as to use the
        coded up analytical expressions for gradients

        Should add creation of instance of QuantumSystemModel here itself instead of it being passed in
        """
        if train_inputs is None:
            T = self.tvec
            Y = self.samples
            U = self.uvec
            W0 = self.w0_vec
            W1 = self.w1_vec
            W2 = self.w2_vec
            M = self.mvec
        else:
            T = train_inputs[:,0]
            Y = train_inputs[:,1]
            U = train_inputs[:,2]
            W0 = train_inputs[:,3]
            W1 = train_inputs[:,4]
            W2 = train_inputs[:,5]
            M = 0*W0 + 1*W1 + 2*W2

        # Compute loss
        N_CONFIG = 2*M + U
        loss_samples = np.zeros(len(Y))
        for ind_sample in range(len(Y)):
            loss_samples[ind_sample] = qs_model.log_likelihood_loss(Y[ind_sample],
                                                                    N_CONFIG[ind_sample],
                                                                    T[ind_sample],
                                                                    FLAG_noise=True)

        loss = np.mean(loss_samples)  # take mean over batch

        # Compute gradient
        grad_analytic = np.zeros(len(param_num))

        for ind_sample in range(len(Y)):
            grad_analytic += qs_model.jacobian_log_likelihood(N_CONFIG[ind_sample], T[ind_sample],
                                                              FLAG_noise=True, type_param=type_param)

        n_samples = len(Y)
        grad_analytic = (1/n_samples)*grad_analytic

        results = {'loss': loss, 'gradient': grad_analytic}
        return results

    def tf_loss(self, param_num, train_inputs=None, type_param='Theta', FLAG_log_sum_exp=True, verbose=True):
        if train_inputs is None:
            T = self.tvec
            Y = self.samples
            U = self.uvec
            W0 = self.w0_vec
            W1 = self.w1_vec
            W2 = self.w2_vec
        else:
            T = train_inputs[:, 0]
            Y = train_inputs[:, 1]
            U = train_inputs[:, 2]
            W0 = train_inputs[:, 3]
            W1 = train_inputs[:, 4]
            W2 = train_inputs[:, 5]

        # model variables
        XI = self.xi_param
        theta_nd = param_num/self.xi_param

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
        if not FLAG_log_sum_exp:
            loss = -1 * tf.math.log((1 - Y) * ((1 - self.r0) * Y_sp + self.r1 * (1 - Y_sp)) +
                                    Y * ((1 - self.r1) * (1 - Y_sp) + self.r0 * Y_sp))  # log likelihood

            loss = tf.reduce_mean(loss)  # take mean over batch
        else:
            r0 = self.r0.astype(np.float32)
            r1 = self.r1.astype(np.float32)
            exp_log_Y = tf.math.exp(tf.math.log(Y))
            # max_exp_log_Y = np.amax(exp_log_Y)
            loss = (1 - exp_log_Y) * ((1 - r0) * tf.cast(Y_sp, tf.float32) + r1 * (1 - tf.cast(Y_sp, tf.float32))) + \
                   exp_log_Y * ((1 - r1) * (1 - tf.cast(Y_sp, tf.float32)) + r0 * tf.cast(Y_sp, tf.float32)) # likelihood
            loss = -1 * tf.math.log(loss)
            loss = tf.reduce_mean(loss)  # take mean over batch

        return loss

    # def tf_train_step(self, optimizer, param_num, train_inputs, type_param, verbose=False):
    #     with tf.GradientTape() as tape:
    #         loss = self.tf_loss(param_num, train_inputs=train_inputs, type_param=type_param)
    #
    #     gradients = tape.gradient(loss, param_num)
    #     optimizer.apply_gradients(zip(gradients, param_num))
    #
    #     return loss

    def tf_test_batch(self, init_param_nd, type_param='Theta', FLAG_log_sum_exp=True, verbose=True, do_plot=False):
        '''
        Snippet of code to look at the values of loss and gradients over batches
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

        # trainable model variables
        init_param_num = init_param_nd * self.xi_param
        param_num = tf.Variable(init_param_num, trainable=True, name="param_num", dtype=tf.float32)

        if verbose:
            print("Input data xy shape=%s" % str(xy.shape))
            print("Each epoch has %d mini_batches of size %s" % (n_mini_batches, mini_batch_size))
            print("Input data shapes samples=%s, tpts=%s" % (self.samples.shape, self.tvec.shape))
            print("initial values: theta=%s" % (param_num,))

        if verbose:
            print("Running MLE over %s datapoints with %s epochs" % (batch_size, nepochs))

        gradients_array = np.zeros(shape=(6, n_mini_batches))
        gradients_trick_array = np.zeros(shape=(6, n_mini_batches))
        loss_array = np.zeros(n_mini_batches)
        summary_queries_array = np.zeros(shape=(4, n_mini_batches))

        np.random.shuffle(xy)  # random in-place permutation of first dimension
        for n in range(n_mini_batches):
            n0 = n * mini_batch_size
            train_batch_temp = xy[n0:n0 + mini_batch_size, :]

            with tf.GradientTape() as tape:
                loss = self.tf_loss(param_num, train_inputs=train_batch_temp,
                                    type_param=type_param, FLAG_log_sum_exp=False, verbose=verbose)

            gradients = tape.gradient(loss, param_num)
            gradients_array[:, n] = gradients.numpy()
            loss_array[n] = loss.numpy()

            with tf.GradientTape() as tape:
                loss_trick = self.tf_loss(param_num, train_inputs=train_batch_temp,
                                    type_param=type_param, FLAG_log_sum_exp=True, verbose=verbose)

            gradients_trick = tape.gradient(loss_trick, param_num)
            gradients_trick_array[:, n] = gradients_trick.numpy()

            # (Indirect) summary of queries being used over each batch
            summary_queries_array[0,n] = np.sum(train_batch_temp[:,2]) # u
            summary_queries_array[1,n] = np.sum(train_batch_temp[:,3]) # w0
            summary_queries_array[2,n] = np.sum(train_batch_temp[:,4]) # w1
            summary_queries_array[3,n] = np.sum(train_batch_temp[:,5]) # w2

        # Final result
        loss = self.tf_loss(param_num, type_param=type_param, verbose=verbose)

        if do_plot and n_mini_batches > 1:
            # For plotting
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = "sans-serif"
            plt.rcParams['font.sans-serif'] = "Helvetica"
            plt.rcParams['figure.titlesize'] = 24
            plt.rcParams['axes.labelsize'] = 24
            plt.rcParams['axes.titlesize'] = 24
            plt.rcParams['legend.fontsize'] = 24
            plt.rcParams['xtick.labelsize'] = 24
            plt.rcParams['ytick.labelsize'] = 24

            plt.figure(1, figsize=(6, 6))
            plt.hist(loss_array, bins=int(np.round(n_mini_batches / 2)))
            plt.tight_layout()
            plt.savefig('loss_batches.png', bbox_inches='tight')
            plt.show()

            plt.figure(2, figsize=(12, 12))
            subplot_indices = [221, 222, 223, 224]
            xlabel_array = ['u', 'w0', 'w1', 'w2']

            for ind in range(4):
                plt.subplot(subplot_indices[ind])
                plt.hist(summary_queries_array[ind, :], bins=int(np.round(n_mini_batches / 2)))
                plt.xlabel(xlabel_array[ind])

            plt.tight_layout()
            plt.savefig('queries_batches.png', bbox_inches='tight')
            plt.show()

            plt.figure(3, figsize=(12,18))
            subplot_indices = [321, 323, 325, 322, 324, 326]
            xlabel_array = [r'$J_{IX}$ ($10^8$)', r'$J_{IY}$ ($10^8$)', r'$J_{IZ}$ ($10^8$)',
                            r'$J_{ZX}$ ($10^8$)', r'$J_{ZY}$ ($10^8$)', r'$J_{ZZ}$ ($10^8$)']

            for ind in range(6):
                plt.subplot(subplot_indices[ind])
                plt.hist(gradients_array[ind,:], bins=int(np.round(n_mini_batches/2)))
                plt.xlabel(xlabel_array[ind])

            plt.tight_layout()
            plt.savefig('gradients_batches.png', bbox_inches='tight')
            plt.show()

            plt.figure(4, figsize=(12, 18))
            subplot_indices = [321, 323, 325, 322, 324, 326]
            xlabel_array = [r'$J_{IX}$ ($10^8$)', r'$J_{IY}$ ($10^8$)', r'$J_{IZ}$ ($10^8$)',
                            r'$J_{ZX}$ ($10^8$)', r'$J_{ZY}$ ($10^8$)', r'$J_{ZZ}$ ($10^8$)']

            for ind in range(6):
                plt.subplot(subplot_indices[ind])
                plt.hist(gradients_trick_array[ind, :], bins=int(np.round(n_mini_batches / 2)))
                plt.xlabel(xlabel_array[ind])

            plt.tight_layout()
            plt.savefig('gradients_trick_batches.png', bbox_inches='tight')
            plt.show()

        ds = {'loss': loss.numpy(), 'results': gradients_array}

        return ds

    def tf_train(self, init_param_nd, type_param='Theta', verbose=True, do_plot=False):
        """
        Generalization for mle_estimation_param for now

        and works only with parametrization of Lambda and assumes binary classifier
        """
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

        # trainable model variables
        init_param_num = init_param_nd * self.xi_param
        param_num = tf.Variable(init_param_num, trainable=True, name="param_num", dtype=tf.float32)

        if verbose:
            print("Input data xy shape=%s" % str(xy.shape))
            print("Each epoch has %d mini_batches of size %s" % (n_mini_batches, mini_batch_size))
            print("Input data shapes samples=%s, tpts=%s" % (self.samples.shape, self.tvec.shape))
            print("initial values: theta=%s" % (param_num,))

        # optimizer
        if self.optimizer == "gd":
            tf_opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            tf_opt_name = "SGD"
        elif self.optimizer == "adagrad":
            tf_opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
            tf_opt_name = "Adagrad"
        else:
            tf_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            tf_opt_name = "Adam"

        losses = []  # record losses at each epoch step
        steps = []

        # run MLE
        if verbose:
            print("Using %s optimmizer, learning rate=%s" % (tf_opt_name, learning_rate))
            print("Running MLE over %s datapoints with %s epochs" % (batch_size, nepochs))

        for k in range(nepochs):
            np.random.shuffle(xy)  # random in-place permutation of first dimension
            for n in range(n_mini_batches):
                n0 = n * mini_batch_size
                train_batch_temp = xy[n0:n0 + mini_batch_size, :]

                with tf.GradientTape() as tape:
                    loss = self.tf_loss(param_num, train_inputs=train_batch_temp,
                                        type_param=type_param, verbose=verbose)

                gradients = tape.gradient(loss, param_num)
                import pdb; pdb.set_trace()

                tf_opt.apply_gradients(zip(gradients, param_num))
                # tf_opt.apply_gradients(gradients, param_num)

            if not (k % neval_period):
                # Evaluate the loss for the entire training dataset
                loss = self.tf_loss(param_num, type_param=type_param, verbose=verbose)
                results_temp = np.array([loss.numpy(), param_num])

                if verbose:
                    print("Epoch %s: loss=%s, theta=%s" % tuple([k] + results_temp))

                losses.append(loss.numpy())
                steps.append(k)

                if np.isnan(loss.numpy()):
                    raise Exception("loss is NaN, quitting!")

        # Final result
        loss = self.tf_loss(param_num, type_param=type_param, verbose=verbose)
        results = np.array([loss.numpy(), param_num.numpy()])

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
                         (self.samples.size, self.optimizer))

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
    """
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    """

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
    """
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    """

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
    """
    One function to rule them all!!!

    Calculates probability using J parameters or
    param_array = (omega0, delta0, phi0, omega1, delta1, phi1)
    :param J_num:
    :param tvec:
    :return:
    """
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


def model_probabilities_tf(J_num, uvec, w0, w1, w2, tvec):
    """
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    """

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
    """
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    """

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
    """
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    """
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
    """
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    """
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
    """
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    """
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


def model_probabilities_decoherence_ibmq_np_nd(J_num_nd, type_param, xi_J, uvec, w0, w1, w2, tvec_expt):
    """
    Same as above but with decoherence incorporated

    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]

    TODO: Generalize; decoherence model hardcoded at the moment
    """
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

    # Time information -- Control noise
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

    # Decoherence
    qs_T1 = 1e-6 * np.array([94.0278, 75.71162])
    qs_T2 = 1e-6 * np.array([177.22575, 128.0758598])

    damping = quantum_device_models.decoherence_two_qubit_model(tvec_expt, qs_T1, qs_T2)

    p0 = damping[0]*p0 + 2*damping[1]
    p1 = damping[0]*p1 + 2*damping[1]
    p2 = damping[0]*p2 + 2*damping[1]

    fpvec = w0 * p0 + w1 * p1 + w2 * p2

    return fpvec


# For new ibm device ibmq_boel
def model_probabilities_J_ibmq_tf_nd(J_num_nd, xi_J, uvec, w0, w1, w2, tvec_expt):
    """
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    """
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
    """
    Return p_um(t) = smcg model probability of measurement outcome 00 for
    preparation u and measurement m at time t. Use analytical formula
    for evaluation and can be used with tensorflow.
    u in [0,1]
    m in [0,1,2]
    """
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
                                             N_0=972, N_batch=486, max_iter=10, FLAG_initial_estimate=True,
                                             verbose=True, do_plot=False, log_file=None):
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
