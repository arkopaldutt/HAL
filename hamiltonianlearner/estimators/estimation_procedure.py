"""
Contains functions for carrying out initial estimation and stage-wise estimation
"""
import numpy as np
import math
import cmath
import scipy.optimize
import scipy.fftpack
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar, curve_fit, leastsq

# HamiltonianLearning -- Oracles, Models and Learners
from ..quantum_system_models import quantum_device_models

# HamiltonianLearning -- Estimators
from . import initial_estimators, fft_estimators, linear_estimators
from . import mle_estimators as mle_estimators


def initial_estimate(A_cr, FLAG_verbose=False, do_plot=False):
    """
    Quick helper function that is an analog of the PointEstimator with defaults

    Inputs:
        A_cr: ActionSpace of Learner (Required)
        FLAG_verbose: prints messages if true
        do_plot: plots the frequency landscape if true

    Returns:
        Parameter estimate based on FFT+LinearRegression
    """
    # Time-stamps and frequency information of ActionSpace
    time_stamps = A_cr.tset * A_cr.xi_t

    if A_cr.freq_convert is None:
        delta_time = (time_stamps[-1] - time_stamps[0]) / (len(time_stamps) - 1)
        sample_freq = 1.0 / delta_time
        freq_convert = sample_freq * 2.0 * np.pi / len(time_stamps)
    else:
        freq_convert = A_cr.freq_convert

    # Calculate the rabi oscillations from the given data
    pvec_data = A_cr.rabi_data()

    # Take fourier transforms of the rabi oscillations
    results_freq = fft_estimators.calc_freq(pvec_data, freq_convert, do_plot=do_plot)
    rabi_freq = results_freq['rabi_freq']

    if FLAG_verbose:
        print('FFT Frequency Estimates')
        print(rabi_freq)

    # Calculate and store the initial frequency fit
    rabi_prefit = np.ndarray(shape=pvec_data.shape, dtype=float)

    for ind_config in range(6):
        cs_array = fft_estimators.cos_sin_features(rabi_freq[(ind_config % 2)], time_stamps)
        reg = LinearRegression().fit(cs_array, pvec_data[ind_config, :])
        rabi_prefit[ind_config] = reg.predict(cs_array)

    # Refine the initial frequency estimates by fitting to the Rabi Oscillations
    # TODO: INTRODUCE TRY AND CATCH WITH 2*freq_convert below
    for ind in [0,1]:
        if rabi_freq[ind] == 0:
            rabi_mid = rabi_freq[ind] + freq_convert/2
        else:
            rabi_mid = rabi_freq[ind]

        res = minimize_scalar(fft_estimators.sumsq_wrt_omega,
                              bracket=(np.amax([rabi_freq[ind] - freq_convert,0]),
                                       rabi_mid,
                                       rabi_freq[ind] + freq_convert),
                              args=(ind, pvec_data, time_stamps))
        rabi_freq[ind] = res.x

        if FLAG_verbose:
            print('Linear Regression Estimates')
            print(rabi_freq)

    # Calculate and store the fitted curves and corresponding Fourier coefficients
    rabi_cos_sin = np.ndarray(shape=pvec_data.shape, dtype=float)
    rabi_residuals = np.ndarray(shape=pvec_data.shape, dtype=float)
    rabi_abc = np.ndarray(shape=(pvec_data.shape[0], 3), dtype=float)

    for ind_config in range(pvec_data.shape[0]):
        cs_array = fft_estimators.cos_sin_features(rabi_freq[ind_config % 2], time_stamps)
        reg = LinearRegression().fit(cs_array, pvec_data[ind_config, :])
        rabi_abc[ind_config, 0] = reg.coef_[0]
        rabi_abc[ind_config, 1] = reg.coef_[1]
        rabi_abc[ind_config, 2] = reg.intercept_
        rabi_cos_sin[ind_config] = reg.predict(cs_array)
        rabi_residuals[ind_config] = pvec_data[ind_config, :] - rabi_cos_sin[ind_config]

    # Calculate initial guesses for gamma, phi, theta
    rabi_gpt = np.ndarray(shape=(2, 3), dtype=float)

    for prep in range(2):
        g0 = -cmath.phase(complex(rabi_abc[4 + prep, 0], rabi_abc[4 + prep, 1]))
        expg0 = cmath.exp(complex(0, g0))
        x0 = expg0 * complex(rabi_abc[0 + prep, 0], rabi_abc[0 + prep, 1])
        y0 = expg0 * complex(rabi_abc[2 + prep, 0], rabi_abc[2 + prep, 1])
        z0 = expg0 * complex(rabi_abc[4 + prep, 0], rabi_abc[4 + prep, 1])
        rabi_gpt[prep, 0] = g0
        rabi_gpt[prep, 1] = cmath.phase(complex(-y0.imag, x0.imag))

        if z0.real > 1.0:
            rabi_gpt[prep, 2] = 0.25 * np.pi
        else:
            rabi_gpt[prep, 2] = (0.25 * np.pi) + 0.5 * np.sign(x0.real * y0.imag) * math.acos(z0.real)

    rabi_refined_gpt = np.ndarray(shape=(2, 3), dtype=float)

    res = scipy.optimize.minimize(linear_estimators.sumsq_wrt_gpt, rabi_gpt[0, :],
                                  args=(0, pvec_data, rabi_freq, time_stamps),
                                  method="trust-constr",
                                  jac=False,  # grad_sumsq_wrt_gpt,
                                  bounds=((-math.pi, math.pi),
                                          (-math.pi, math.pi),
                                          (0, 0.5 * math.pi))
                                  )
    rabi_refined_gpt[0, :] = res.x

    res = scipy.optimize.minimize(linear_estimators.sumsq_wrt_gpt, rabi_gpt[1, :],
                                  args=(1, pvec_data, rabi_freq, time_stamps),
                                  method="trust-constr",
                                  jac=False,  # grad_sumsq_wrt_gpt,
                                  bounds=((-math.pi, math.pi),
                                          (-math.pi, math.pi),
                                          (0, 0.5 * math.pi))
                                  )
    rabi_refined_gpt[1, :] = res.x

    return rabi_freq, rabi_refined_gpt


def baseline_estimate(data, A_cr):
    """
    Get estimate of J using the baseline solver of FFT + Regression
    """
    initial_est = initial_estimators.PointEstimator(A_cr)

    #param_array_IC = initial_est.baseline()
    param_array_IC = initial_est.point_estimator()

    # Get MLE loss of estimate and return
    J_num = quantum_device_models.transform_theta_to_J(param_array_IC)
    mle_est = mle_estimators.MLE_Estimator(data, A_cr.xi_J)
    loss_J = mle_est.np_loss(J_num, type_param='J')

    return loss_J, J_num


def quick_mle_estimate(data, A_cr, xi_J_est=None, init_J=None, FLAG_track_cost=False, FLAG_verbose=False):
    """
    Quick helper function to get an MLE estimate according to the default solver options

    Note that the initial estimation is FFT + Refinement + Regression and not simply a baseline solve

    Inputs:
        data: dataset
        A_cr: ActionSpace
        xi_J_est: normalization factors for Hamiltonian parameters used for estimation

    Returns:
        negative log-likelihood loss, estimate
    """
    if xi_J_est is None:
        xi_J_est = A_cr.xi_J

    # Initial estimation
    if init_J is None:
        if FLAG_verbose:
            print("FFT + Regression for initial estimation")

        initial_est = initial_estimators.PointEstimator(A_cr)
        param_array_IC = initial_est.point_estimator()
        J_num_IC = quantum_device_models.transform_theta_to_J(param_array_IC)
    else:
        if FLAG_verbose:
            print("Initial estimate provided")

        J_num_IC = np.copy(init_J)

    # MLE -- Quasi-Newton Method
    J_num_IC_nd = J_num_IC/xi_J_est
    mle_est = mle_estimators.MLE_Estimator(data, xi_J_est)

    if FLAG_track_cost:
        J_num_nd, cost = mle_est.quasi_newton_solve(J_num_IC_nd, FLAG_track_cost=FLAG_track_cost)
    else:
        J_num_nd = mle_est.quasi_newton_solve(J_num_IC_nd)

    J_num = J_num_nd*xi_J_est

    loss_J = mle_est.np_loss(J_num)

    if FLAG_track_cost:
        return loss_J, J_num, cost
    else:
        return loss_J, J_num


# Main - Complete estimation procedure
def estimation_procedure(data, init_J=None, env_cr=None, A_cr=None,
                         estimation_strategy=None,
                         solver_options=None,
                         verbose=False, verbose_local=False, do_plot=False):
    """
    Get estimate of J
    1. Run initial estimation procedure -- output is in parameterization Theta
    2. Run SGD using Lambda/Theta parameterization
    3. Run SDG using J parameterization
    4. Compare all estimates
    5. Return result in J parameterization
    """
    # Set up the structure for J_num and corresponding loss
    J_num_array = np.zeros((3,6))
    loss_array = np.zeros(3)

    if estimation_strategy is None:
        estimation_strategy = {'FLAG_initial_estimate': True, 'FLAG_FFT_high_resolution': True,
                                       'FLAG_MLE_param': True, 'FLAG_MLE_J': True}

    if estimation_strategy['FLAG_initial_estimate']:
        if estimation_strategy['FLAG_FFT_high_resolution']:
            initial_est = initial_estimators.InitialEstimator(data, env_cr=env_cr, A_cr=A_cr,
                                                              FLAG_mle_solver=False,
                                                              FLAG_fft_window=True, type_fft_window='hamming',
                                                              FLAG_fft_high_resoultion=True)

            param_array_IC = initial_est.point_estimator_linear_regression(freq_resolution=4)
        else:
            # Old code based on initial_estimate
            rabi_freq, rabi_gpt = initial_estimate(A_cr=A_cr, FLAG_verbose=verbose, do_plot=do_plot)
            param_array_IC = np.array([rabi_freq[0] / 2, 2 * (rabi_gpt[0, 2] - np.pi / 4), rabi_gpt[0, 1],
                                       rabi_freq[1] / 2, 2 * (rabi_gpt[1, 2] - np.pi / 4), rabi_gpt[1, 1]])

        if verbose_local:
            print(param_array_IC)

        # except:
        #     if verbose_local:
        #         print('Initial Estimation Failed! Using previous value of J.')
        #
        #     param_array_IC = transform_parameters(init_J)
    else:
        if init_J is None:
            raise RuntimeError('You cant skip initial estimation if you do not pass init_J as argument')

        param_array_IC = quantum_device_models.transform_parameters(init_J)

    # Set up the MLE Estimator
    J_temp = quantum_device_models.transform_theta_to_J(param_array_IC)
    J_num_array[0,:] = np.copy(J_temp)

    xi_J_temp = 10 ** (np.floor(np.log10(np.abs(J_temp))) + 1)
    # xi_param = np.array([np.amax(xi_J_temp), 1, 1, np.amax(xi_J_temp), 1, 1])
    # xi_param = 10 ** (np.log10(np.abs(param_array_IC)).astype(int) + 1)
    xi_param = 10 ** (np.floor(np.log10(np.abs(param_array_IC))) + 1)

    mle_est = mle_estimators.MLE_Estimator(data, xi_param, solver_options=solver_options)
    loss_array[0] = mle_est.np_loss(param_array_IC, type_param = 'Theta')

    if estimation_strategy['FLAG_MLE_param']:
        # Compare the log-likelihood loss of param_array_IC and that from init_J and choose one as the x0 for SGD
        if init_J is not None:
            if mle_est.np_loss(param_array_IC, verbose=verbose) > mle_est.np_loss(transform_parameters(init_J), verbose=verbose):
                if verbose_local:
                    print('Previous value of J has better loss')

                param_array_IC = quantum_device_models.transform_parameters(init_J)

                # Reset the mle_estimator now
                xi_J_temp = 10 ** (np.floor(np.log10(np.abs(init_J))) + 1)
                xi_param = 10 ** (np.floor(np.log10(np.abs(param_array_IC))) + 1)

                mle_est = mle_estimators.MLE_Estimator(data, xi_param, solver_options=solver_options)

        param_nd0 = param_array_IC / xi_param
        ds = mle_est.tf_sgd(param_nd0, verbose=verbose, do_plot=do_plot)

        loss_param = ds['results'][0]
        param_array_nd = ds['results'][1]
        param_array_num = param_array_nd * xi_param
        J_num_array[1,:] = np.copy(quantum_device_models.transform_theta_to_J(param_array_num))
        loss_array[1] = np.copy(loss_param)
    else:
        J_num_array[1,:] = np.copy(J_num_array[0,:])
        loss_array[1] = np.copy(loss_array[0])

    # Run estimation using the J parameterization
    J_num0 = np.copy(J_num_array[1,:])
    if estimation_strategy['FLAG_MLE_J']:
        try:
            if solver_options is not None:
                solver_options = {'mini_batch_size': solver_options['mini_batch_size'], 'nepochs': 250,
                                  'neval_period': 50, 'optimizer': 'adam', 'learning_rate': 0.001}
            else:
                solver_options = {'nepochs': 250, 'neval_period': 50, 'optimizer': 'adam', 'learning_rate': 0.001}

            xi_J = 10**(np.floor(np.log10(np.abs(J_num0))) + 1)
            mle_est = mle_estimators.MLE_Estimator(data, xi_J, solver_options=solver_options)

            J_nd0 = J_num0/xi_J
            ds = mle_est.tf_sgd(J_nd0, type_param='J', verbose=verbose, do_plot=do_plot)

            loss_J = ds['results'][0]
            J_nd = ds['results'][1]
            J_num1 = J_nd * xi_J

            J_num_array[2,:] = np.copy(J_num1)
            loss_array[2] = loss_J
        except:
            if verbose_local:
                J_num_array[2, :] = np.copy(J_num_array[1,:])
                loss_array[2] = np.copy(loss_array[1])
                print('MLE Solve in J Parameterization Failed')
    else:
        J_num_array[2, :] = np.copy(J_num_array[1, :])
        loss_array[2] = np.copy(loss_array[1])

    return np.amin(loss_array), J_num_array[np.argmin(loss_array),:]


def normalized_L2_error(J_hat, J_num, xi_J):
    return np.linalg.norm((J_num - J_hat)/xi_J, 2)
