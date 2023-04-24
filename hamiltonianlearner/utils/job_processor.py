"""
File containing functions to post process results from jobs
"""
# Add paths
import os
from typing import Dict, List, Any, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pickle

# package imports
from ..quantum_system_models import quantum_device_models
from ..estimators.estimation_procedure import normalized_L2_error
from ..estimators.mle_estimators import MLE_Estimator


def as_si(x, ndp):
    """
    Quick utility function to help with annotation of texts on figures
    Ref: https://stackoverflow.com/questions/31453422/displaying-numbers-with-x-instead-of-e-scientific-notation-in-matplotlib/31453961
    """
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


def extract_results_log(contents_log, max_iters, max_runs, FLAG_log_mod=False, N_0=972, N_batch=100,
                        FLAG_filter=True, factor_filter=1.9, ind_iter_filter=[0],
                        do_plot=True, do_plot_filter=False, save_filename='results.png',
                        label_learners=['Uniform']):

    n_rows_log = contents_log.shape[0]
    print(n_rows_log % max_iters)
    if n_rows_log % max_iters != 0:
        raise RuntimeError('Check maximum iteration number')

    n_runs = int(n_rows_log/max_iters)
    print('%d/%d runs failed' % (max_runs-n_runs, max_runs) )

    # Let's reshape the results of the contents to check I am reading the right things
    if FLAG_log_mod:
        iters_job = np.reshape(contents_log[:, 0], [n_runs, max_iters])
        rmse_job = np.reshape(contents_log[:, 1], [n_runs, max_iters])
        test_error_job = np.reshape(contents_log[:, 2], [n_runs, max_iters])
    else:
        runs_job = np.reshape(contents_log[:, 0], [n_runs, max_iters])
        iters_job = np.reshape(contents_log[:, 1], [n_runs, max_iters])
        rmse_job = np.reshape(contents_log[:, 2], [n_runs, max_iters])
        test_error_job = np.reshape(contents_log[:, 3], [n_runs, max_iters])

    print(iters_job[0,-1])
    if max_iters != (iters_job[0,-1] + 1):
        raise RuntimeError('Max iters not equal to what is in the log!')

    # Filter out the bad runs -- we can do this by removing the outliers of the rmse, test error or distance of J from mean
    # We choose testing error as this is closest to what we would do in practice
    # RMSE is not available in practice because we wouldn't have access to the truth
    if FLAG_filter:
        for i_iter in ind_iter_filter:

            if do_plot_filter:
                # Let's probe the histogram of the first iteration
                plt.figure(0, figsize=(8,8))
                plt.hist(test_error_job[:, i_iter], bins=int(n_runs/2))
                plt.xlabel('Negative Log-Likelihood Loss')
                plt.ylabel('Count')

                plt.figure(1, figsize=(8, 8))
                plt.hist(rmse_job[:, i_iter], bins=int(n_runs/2))
                plt.xlabel('RMSE')
                plt.ylabel('Count')

            # Remove the outliers using the first iteration
            test_mean = np.mean(test_error_job, axis=0)
            test_std = np.std(test_error_job, axis=0)

            rmse_mean = np.mean(rmse_job, axis=0)
            rmse_std = np.std(rmse_job, axis=0)

            test_runs_iter = np.abs(test_error_job[:,i_iter] - test_mean[i_iter])
            rmse_runs_iter = np.abs(rmse_job[:, i_iter] - rmse_mean[i_iter])

            outlier_run_ids = np.where(test_runs_iter > factor_filter*test_std[i_iter])[0].tolist()
            outlier_run_ids2 = np.where(rmse_runs_iter > factor_filter * rmse_std[i_iter])[0].tolist()

            if not FLAG_log_mod:
                print('Outliers (RUN IDS) in log-likelihood loss are:')
                print(runs_job[outlier_run_ids,0])

                print('Outliers (RUN IDS) in RMSE are:')
                print(runs_job[outlier_run_ids2, 0])

            if not FLAG_log_mod:
                runs_job = np.delete(runs_job, outlier_run_ids, axis=0)

            iters_job = np.delete(iters_job, outlier_run_ids, axis=0)
            rmse_job = np.delete(rmse_job, outlier_run_ids, axis=0)
            test_error_job = np.delete(test_error_job, outlier_run_ids, axis=0)

            n_runs = n_runs - len(outlier_run_ids)

    # Get the mean results now
    test_mean = np.mean(test_error_job, axis=0)
    test_std = np.std(test_error_job, axis=0)

    rmse_mean = np.mean(rmse_job, axis=0)
    rmse_std = np.std(rmse_job, axis=0)

    rmse_errorbars = np.zeros((2,max_iters))
    for ind in range(max_iters):
        rmse_errorbars[0, ind] = np.abs(np.amin(rmse_job[:,ind]) - rmse_mean[ind])
        rmse_errorbars[1, ind] = np.abs(np.amax(rmse_job[:, ind]) - rmse_mean[ind])

    # Slopes
    N_p = N_0 + np.arange(max_iters) * N_batch
    slope_end = 0
    for i_run in range(n_runs):
        rmse_i = rmse_job[i_run,:]
        poly_end = np.polyfit(np.log(N_p[-6:]), np.log(rmse_i[-6:]), 1)
        slope_end += poly_end[0]

    slope_end = slope_end/n_runs
    print(slope_end)

    if do_plot:
        plotting_options = {'label_learners': label_learners, 'save_filename': save_filename}
        plot_trend([rmse_mean], [N_p], error_learners=[rmse_std], plotting_options=plotting_options)

    return {'rmse_mean': rmse_mean, 'rmse_std': rmse_std, 'rmse_errorbars': rmse_errorbars,
            'test_mean': test_mean, 'test_std': test_std, 'N_p': N_p}


def read_results_log(contents_log, n_runs, n_iters, N_0=972, N_batch=100):
    L2_error = np.zeros(n_iters)
    test_error = np.zeros(n_iters)
    N_p = N_0 + np.arange(n_iters) * N_batch

    for i_run in range(n_runs):
        row_start = i_run * n_iters
        row_end = (i_run + 1) * n_iters

        #print(row_start, row_end)
        L2_error += contents_log[row_start:row_end, 2]
        test_error += contents_log[row_start:row_end, 3]

    # Change to RMSE and divide by number of runs
    L2_error = L2_error / n_runs
    test_error = np.sqrt(test_error / N_p)

    return L2_error, test_error, N_p


def plot_trend(rmse_learners, nqueries_learners, nqueries_lower_bound=None,
               rmse_lower_bound=None, error_learners=None, plotting_options=None):

    # Default plotting options
    default_plotting_options = {'label_learners': ['Uniform (Random)', 'AL FIR (Test: p_u)', 'AL FI (Test: Unknown)'],
                                'FLAG_save_plot': True, 'save_filename': 'results.png',
                                'figsize_plot': (12, 10), 'skip_learner': 0,
                                'FLAG_fit_slopes': [True, True],
                                'n_iters_end': None, 'slope_scales': 0.9*np.ones(len(rmse_learners)),
                                'FLAG_reduced_HL': False,
                                'FLAG_expt_data': False,
                                'FLAG_testing_error': False, 'FLAG_long_time_range': False, 'FLAG_legend_outside': True}

    if plotting_options is None:
        plotting_options = default_plotting_options
    else:
        for _key in default_plotting_options.keys():
            plotting_options.setdefault(_key, default_plotting_options[_key])

    # Plot the trends
    if plotting_options['FLAG_expt_data']:
        color_lines = ['r-', 'b-', 'g-', 'm-', 'r--', 'b--', 'g--', 'm--']
        marker_color = ['r', 'b', 'g', 'mediumturquoise', 'hotpink', 'y']
        marker_style = ['o', 'o', 'o', 'o', 'o', 'o']
    else:
        color_lines = ['r-', 'b-', 'g-', 'm-', 'r--', 'b--', 'g--', 'm--']
        marker_color = ['r', 'b', 'g', 'm', 'c', 'y']
        marker_style = ['ro', 'bo', 'go', 'mo', 'co', 'yo']

    # To force xlabels and ylabels to use the right defined Font
    # Ref: https://stackoverflow.com/questions/11611374/sans-serif-font-for-axes-tick-labels-with-latex
    from matplotlib import rc
    rc('text.latex', preamble=r'\usepackage{sfmath}')

    plt.figure(figsize=plotting_options['figsize_plot'])
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['text.usetex'] = True

    # change the scaling if testing error plots
    if plotting_options['FLAG_testing_error']:
        factor_rescale_plot = 100
    else:
        factor_rescale_plot = 1

    slope_scales = plotting_options['slope_scales']

    # If we want to use the marker/color information starting from a different counter
    skip_learner = plotting_options['skip_learner']

    # Loop over the list of rmse for different learners
    for ind in range(len(rmse_learners)):
        rmse_ind = factor_rescale_plot*rmse_learners[ind]
        nqueries_ind = nqueries_learners[ind]

        # Plotting styles, labels, etc.
        color_learner_plot = marker_color[ind + skip_learner]
        marker_style_plot = marker_style[ind + skip_learner]
        label_learner = plotting_options['label_learners'][ind]

        if hasattr(plotting_options['n_iters_end'], '__len__'):
            # Ref: https://stackoverflow.com/questions/23567628/how-to-check-if-len-is-valid
            if len(plotting_options['n_iters_end']) > 1:
                n_iters_end = plotting_options['n_iters_end'][ind]
        else:
            n_iters_end = plotting_options['n_iters_end']

        if n_iters_end is None:
            n_iters_end = round(len(nqueries_ind) / 3)

        # RMSE of Learners (Data) with RMSE Errorbars if available
        if error_learners is not None:
            error_ind = factor_rescale_plot*error_learners[ind]

            if error_ind.shape[0] == 2:
                plt.fill_between(nqueries_ind, rmse_ind-error_ind[0,:], rmse_ind+error_ind[1,:],
                                 color=color_learner_plot, alpha=0.2)
            else:
                plt.fill_between(nqueries_ind, rmse_ind - error_ind, rmse_ind + error_ind,
                                 color=color_learner_plot, alpha=0.2)

            # Errorbar might not be working
            # plt.errorbar(nqueries_ind, rmse_ind, yerr=error_ind,
            #              fmt=marker_style[ind], mec=marker_color[ind], uplims=True, lolims=True)

        plt.plot(nqueries_ind, rmse_ind, marker_style_plot, c=color_learner_plot, label=label_learner)

        # RMSE of Learners (Fits) -- beginning and ending of trends
        if plotting_options['FLAG_fit_slopes'][0]:
            poly_init, cov_init = np.polyfit(np.log(nqueries_ind[0:6]), np.log(rmse_ind[0:6]), 1, cov=True)
            label_fit_init = r"slope = %0.2f $\pm$ %0.2f" % (poly_init[0], np.sqrt(np.diag(cov_init)[0]))

            plt.plot(nqueries_ind[0:6], np.exp(np.polyval(poly_init,
                                                          np.log(nqueries_ind[0:6]))),
                     color=color_learner_plot, linestyle='dashed', linewidth=2, label=label_fit_init)

        if plotting_options['FLAG_fit_slopes'][1]:
            poly_end, cov_end = np.polyfit(np.log(nqueries_ind[-n_iters_end:]),
                                           np.log(rmse_ind[-n_iters_end:]), 1, cov=True)

            label_fit_end = r"slope = %0.2f $\pm$ %0.2f" % (poly_end[0], np.sqrt(np.diag(cov_end)[0]))

            plt.plot(nqueries_ind[-n_iters_end:],
                     slope_scales[ind]*np.exp(np.polyval(poly_end, np.log(nqueries_ind[-n_iters_end:]))),
                     color=color_learner_plot, linestyle='-', linewidth=2, label=label_fit_end)

        # poly_all, cov_all = np.polyfit(np.log(nqueries_ind), np.log(rmse_ind), 1, cov=True)
        # label_fit_all = r"Slope = %0.2f $\pm$ %0.2f" % (poly_all[0], np.sqrt(np.diag(cov_all)[0]))
        # plt.plot(nqueries_ind, np.exp(np.polyval(poly_all, np.log(nqueries_ind))),
        #          'k:', label=label_fit_all)

    if rmse_lower_bound is not None and nqueries_lower_bound is not None:
        plt.plot(nqueries_lower_bound, rmse_lower_bound, 'k-', label='Cramer-Rao Bound')
        poly_crb, cov_crb = np.polyfit(np.log(nqueries_lower_bound), np.log(rmse_lower_bound), 1, cov=True)
        print(poly_crb)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Number of queries")

    if not plotting_options['FLAG_reduced_HL']:
        if plotting_options['FLAG_testing_error']:
            plt.ylabel(r"Testing Error ($\times 10^{-2}$)")

            if plotting_options['FLAG_long_time_range']:
                # Long time-range (FIR)
                plt.xticks([7e3, 1e4, 2e4, 5e4, 1e5])
                plt.gca().set_xticklabels([r'$7 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$', r'$5 \times 10^{4}$',
                                           r'$10^{5}$'])

                # plt.yticks([1.0e-3, 5.0e-4, 2.0e-4, 1.0e-4, 5.0e-5, 1.0e-5])
                # plt.gca().set_yticklabels([r'$10^{-3}$', r'$5 \times 10^{-4}$', r'$2 \times 10^{-4}$',
                #                            r'$10^{-4}$', r'$5 \times 10^{-4}$', r'$10^{-5}$'])
            else:
                # Short time-range (FI)
                plt.xticks([2e3,5e3,1e4,2e4,5e4,1e5])
                plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$',
                                           r'$5 \times 10^{4}$', r'$10^{5}$'])

                # plt.yticks([2e-2,5e-2,1e-1,2e-1,5e-1])
                # plt.gca().set_yticklabels([r'$2 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$',
                #                            r'$2 \times 10^{-1}$', r'$5 \times 10^{-1}$'])
        else:
            plt.ylabel("RMSE")
            if plotting_options['FLAG_long_time_range']:
                # Long time-range (FI)

                # Limits
                plt.ylim([2e-2, 0.6])

                plt.xticks([7e3,1e4,2e4,5e4,1e5])
                plt.gca().set_xticklabels([r'$7 \times 10^{3}$', r'$10^{4}$',
                                           r'$2 \times 10^{4}$', r'$5 \times 10^{4}$',
                                           r'$10^{5}$'])

                plt.yticks([2e-2,5e-2,1e-1,2e-1,5e-1])
                plt.gca().set_yticklabels([r'$2 \times 10^{-2}$',
                                           r'$5 \times 10^{-2}$',
                                           r'$10^{-1}$',
                                           r'$2 \times 10^{-1}$',
                                           r'$5 \times 10^{-1}$'])
            else:
                # Short time-range (FI)

                # Limits
                plt.ylim([0.5e-2, 2.5])

                # Tick-marks
                plt.xticks([2e3,5e3,1e4,2e4,5e4,1e5])
                plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$',
                                           r'$5 \times 10^{4}$', r'$10^{5}$'])

                plt.yticks([2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0])
                plt.gca().set_yticklabels([r'$2 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$',
                                           r'$2 \times 10^{-1}$', r'$5 \times 10^{-1}$', r'$10^{0}$', r'$2 \times 10^{0}$'])
    else:
        plt.ylim([1.0e-4, 2])
        plt.ylabel("RMSE")
        if plotting_options['FLAG_long_time_range']:
            plt.xticks([2e3, 5e3, 1e4, 2e4, 5e4])
            plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$',
                                       r'$5 \times 10^{4}$'])
        else:
            plt.xticks([2e3, 5e3, 1e4, 2e4])
            plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$'])

    if plotting_options['FLAG_expt_data']:
        plt.legend(loc='lower left', ncol=2)
    else:
        if plotting_options['FLAG_legend_outside']:
            plt.legend(bbox_to_anchor=(1.04, 1))
        else:
            #plt.legend(loc="upper right")
            plt.legend(loc='best')

    if plotting_options['FLAG_save_plot']:
        plt.savefig(plotting_options['save_filename'], bbox_inches='tight', dpi=600)


def extract_run_data(SAVE_DIR, run_id, i_iter):
    pickle_run_file = SAVE_DIR + '/Run_%d.pickle' % run_id
    results_PL = pickle.load(open(pickle_run_file, "rb"))

    PL_data = results_PL['data']
    N_p_iter = results_PL['N_p'][i_iter]

    PL_data_iter = {'xi_t': PL_data['xi_t'], 'time_stamps': PL_data['time_stamps'],
                    'freq_convert': PL_data['freq_convert'], 'FLAG_classification': PL_data['FLAG_classification'],
                    'misclassif_error': PL_data['misclassif_error'], 'device': PL_data['device'],
                    'xi_J': PL_data['xi_J'], 'samples': PL_data['samples'][0:N_p_iter],
                    'mvec': PL_data['mvec'][0:N_p_iter],
                    'uvec': PL_data['uvec'][0:N_p_iter], 'tvec': PL_data['tvec'][0:N_p_iter],
                    'time': PL_data['time'][0:N_p_iter],
                    'config': PL_data['config'][0:N_p_iter], 'actions': PL_data['actions'][0:N_p_iter]}

    return results_PL, PL_data_iter, N_p_iter


def empirical_estimator(SAVE_DIR, n_iters, max_iters=125, xi_J=np.array([1e6 for i in range(6)]),
                        FLAG_plot=False, label_learners=[r'HAL-FI'], FLAG_diagnostics=False):

    list_run_files = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".pickle")]
    #n_runs = len(list_run_files)

    run_file = list_run_files[0]
    with open(run_file, "rb") as f_run:
        results_run = pickle.load(f_run)

    N_actions_temp = len(results_run['q'][-1])
    q_vec = np.zeros(N_actions_temp)

    J_runs_vec = []
    theta_runs_vec = []
    theta_temp = np.zeros(shape=(n_iters,6))
    valid_runs_vec = []

    n_runs = 0
    counter = 0
    for run_file in list_run_files:
        with open(run_file, "rb") as f_run:
            results_run = pickle.load(f_run)

        if len(results_run['J_hat']) >= max_iters:
            # TODO: Need to update this according to the way we report the filtered q distribution now
            # if 'q' in results_run.keys():
            #     q_vec += results_run['q'][-1]

            J_runs_vec.append(np.array(results_run['J_hat'])[0:max_iters])

            for ind_iter in range(max_iters):
                theta_temp[ind_iter] = quantum_device_models.transform_parameters(results_run['J_hat'][ind_iter])

            theta_runs_vec.append(np.copy(theta_temp))
            n_runs += 1
            valid_runs_vec.append(counter)

        counter += 1

    # Load the last valid run and get the trend of number of queries with iterations
    run_file = list_run_files[valid_runs_vec[-1]]
    with open(run_file, "rb") as f_run:
        results_run = pickle.load(f_run)

    N_p = results_run['N_p'][0:max_iters]

    # Take average
    q_vec = q_vec / n_runs

    # Get empirical rmse
    rmse_J_i_iter_vec = np.zeros(shape=(n_iters, n_runs, 6))
    rmse_J_i_trend = np.zeros(shape=(n_iters,6))

    rmse_theta_i_iter_vec = np.zeros(shape=(n_iters, n_runs, 6))
    rmse_theta_i_trend = np.zeros(shape=(n_iters, 6))

    rmse_J_iter_vec = np.zeros(shape=(n_iters, n_runs))
    rmse_J_trend = np.zeros(n_iters)
    J_mean = np.zeros(shape=(n_iters,6))

    rmse_theta_iter_vec = np.zeros(shape=(n_iters, n_runs))
    rmse_theta_trend = np.zeros(n_iters)
    theta_mean = np.zeros(shape=(n_iters, 6))

    for ind_run in range(n_runs):
        J_mean += J_runs_vec[ind_run]
        theta_mean += theta_runs_vec[ind_run]

    J_mean = J_mean/n_runs
    theta_mean = theta_mean / n_runs

    xi_theta = np.array([1e6,1,1,1e6,1,1])
    for ind_run in range(n_runs):
        J_temp_run = J_runs_vec[ind_run]
        theta_temp_run = theta_runs_vec[ind_run]
        for ind_iter in range(n_iters):
            rmse_J_trend[ind_iter] += normalized_L2_error(J_temp_run[ind_iter], J_mean[ind_iter], xi_J) ** 2
            rmse_J_iter_vec[ind_iter, ind_run] = normalized_L2_error(J_temp_run[ind_iter], J_mean[ind_iter], xi_J)

            rmse_theta_trend[ind_iter] += normalized_L2_error(theta_temp_run[ind_iter],
                                                                         theta_mean[ind_iter], xi_theta) ** 2

            rmse_theta_iter_vec[ind_iter, ind_run] = normalized_L2_error(theta_temp_run[ind_iter],
                                                                         theta_mean[ind_iter], xi_theta)

            for ind_component in range(6):
                rmse_J_i_trend[ind_iter, ind_component] += \
                    ((J_temp_run[ind_iter, ind_component] - J_mean[ind_iter, ind_component])/xi_J[ind_component])**2
                rmse_J_i_iter_vec[ind_iter, ind_run, ind_component] = \
                    ((J_temp_run[ind_iter, ind_component] - J_mean[ind_iter, ind_component])/xi_J[ind_component])**2

                rmse_theta_i_trend[ind_iter, ind_component] += \
                    ((theta_temp_run[ind_iter, ind_component] - theta_mean[ind_iter, ind_component]) / xi_theta[ind_component]) ** 2
                rmse_theta_i_iter_vec[ind_iter, ind_run, ind_component] = \
                    ((theta_temp_run[ind_iter, ind_component] - theta_mean[ind_iter, ind_component]) / xi_theta[ind_component]) ** 2

    rmse_J_trend = np.sqrt(rmse_J_trend/n_runs)
    rmse_J_i_trend = np.sqrt(rmse_J_i_trend / n_runs)
    rmse_J_std = np.std(rmse_J_iter_vec, axis=1)

    rmse_theta_trend = np.sqrt(rmse_theta_trend / n_runs)
    rmse_theta_i_trend = np.sqrt(rmse_theta_i_trend / n_runs)
    rmse_theta_std = np.std(rmse_theta_iter_vec, axis=1)

    if FLAG_plot:
        plot_trend([rmse_J_trend], [N_p], error_learners=[rmse_J_std], label_learners=label_learners,
                   save_filename='rmse_J_trend.png')
        plot_trend([rmse_theta_trend], [N_p], error_learners=[rmse_theta_std], label_learners=label_learners,
                   save_filename='rmse_theta_trend.png')

    if FLAG_diagnostics:
        rmse_J_temp = []
        rmse_theta_temp = []
        N_temp = []
        label_J_temp = ['Jix', 'Jiy', 'Jiz', 'Jzx', 'Jzy', 'Jzz']
        label_theta_temp = [r'$\omega_0$', r'$\delta_0$', r'$\phi_0$', r'$\omega_1$', r'$\delta_1$', r'$\phi_1$']
        for ind in range(6):
            rmse_J_temp.append(rmse_J_i_trend[:,ind])
            rmse_theta_temp.append(rmse_theta_i_trend[:, ind])

            N_temp.append(N_p)

        if FLAG_plot:
            plot_trend(rmse_J_temp, N_temp, label_learners=label_J_temp,
                       save_filename='rmse_J_components_trend.png')
            plot_trend(rmse_theta_temp, N_temp, label_learners=label_theta_temp,
                       save_filename='rmse_theta_components_trend.png')

    return rmse_J_trend, q_vec, rmse_J_std


class EmpiricalEstimator(object):
    """
    Class with methods to carry out empirical estimation from log-files and pickle-files over different runs
    """
    def __init__(self, SAVE_DIR, xi_J=np.array([1e6 for _ in range(6)]),
                 max_iters=251, N_0=5*486, N_batch=486,
                 FLAG_long_time_range=False,
                 FLAG_filter=False, FLAG_rmse_test=False, FLAG_param_convert=True,
                 do_plot=False, FLAG_verbose=False):
        """
        Inputs:
            SAVE_DIR:
            max_iters, N_0, N_batch: Usual meaning
            FLAG_mod:
            FLAG_filter:
            FLAG_rmse_test: Did the experiment that we ran involve computation of "Testing" RMSE
        """
        self.SAVE_DIR = SAVE_DIR
        self.xi_J = xi_J
        self.max_iters = max_iters
        self.N_0 = N_0
        self.N_batch = N_batch
        self.FLAG_filter = FLAG_filter

        # Load results from the save directory
        list_run_files = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".txt")]

        # Define RMSE over different runs
        rmse_job = []
        Jhat_job = []
        rmse_test_job = []
        loss_train_job = []     # Training loss

        n_runs = 0
        n_runs_test = 0
        for run_file in list_run_files:
            try:
                contents_run = np.loadtxt(run_file)
            except Exception:
                print('skipping %s' %run_file)
                pass

            if contents_run.shape[0] >= max_iters:
                rmse_run_temp = contents_run[0:max_iters, 1]
                rmse_job.append(rmse_run_temp)

                Jhat_run_temp = contents_run[0:max_iters, 3:9]
                Jhat_job.append(Jhat_run_temp)

                if FLAG_rmse_test:
                    # For some reason some of the log files are not written to properly
                    if contents_run.shape[1] == 10:
                        rmse_test_run_temp = contents_run[0:max_iters, 9]
                        rmse_test_job.append(rmse_test_run_temp)

                        n_runs_test += 1

                n_runs += 1

        if FLAG_verbose:
            print("Loaded %d runs with %d rounds of AL completed!" % (n_runs, max_iters))

        # Save results in a dictionary and update during computation
        self.job_results = {'rmse': np.array(rmse_job), 'Jhat': np.array(Jhat_job),
                            'n_runs': n_runs, 'n_runs_test': n_runs_test, 'max_iters': max_iters}

        # History of number of queries
        N_p = N_0 + np.arange(max_iters) * N_batch
        self.N_p = N_p

        if FLAG_rmse_test:
            self.job_results.update({'rmse_test': np.array(rmse_test_job)})
        else:
            self.job_results.update({'rmse_test': None})

        # For reduced HL problem -- estimation of frequencies
        self.FLAG_param_convert = FLAG_param_convert

        if FLAG_param_convert:
            # Hold all parameters for each run and over all iterations
            param_hat_job = np.zeros(shape=(n_runs, max_iters, 6))

            # Transform all the Jhat into param_hat
            for ind_run in range(n_runs):
                J_temp_run = self.job_results['Jhat'][ind_run, :, :]
                for ind_iter in range(max_iters):
                    J_num_temp = J_temp_run[ind_iter, :]
                    param_num_temp = quantum_device_models.transform_parameters(J_num_temp)

                    param_hat_job[ind_run, ind_iter, :] = param_num_temp

            self.job_results.update({'param_hat': param_hat_job})
        else:
            self.job_results.update({'param_hat': None})

        # For plotting
        self.FLAG_long_time_range = FLAG_long_time_range
        self.do_plot = do_plot

    def filter_outliers(self, rmse_job, n_runs, factor_filter=1.9, ind_iter_filter=[0]):
        """
        Remove outliers from the runs

        We can do this by removing the outliers of the rmse, test error or distance of J from mean
        We choose testing error as this is closest to what we would do in practice

        RMSE is not available in practice because we wouldn't have access to the truth

        Inputs:
            factor_filter:
            ind_iter_filter: The different AL iterations to consider
        """
        for i_iter in ind_iter_filter:
            if self.do_plot:
                # Let's probe the histogram of the first iteration
                plt.figure(1, figsize=(8, 8))
                plt.hist(rmse_job[:, i_iter], bins=int(n_runs / 2))
                plt.xlabel('RMSE')
                plt.ylabel('Count')

            # Remove the outliers using the first iteration
            rmse_mean = np.mean(rmse_job, axis=0)
            rmse_std = np.std(rmse_job, axis=0)

            rmse_runs_iter = np.abs(rmse_job[:, i_iter] - rmse_mean[i_iter])

            outlier_run_ids = np.where(rmse_runs_iter > factor_filter * rmse_std[i_iter])[0].tolist()

            print('Outliers (RUN IDS) in RMSE are:')
            print(outlier_run_ids)

            rmse_job = np.delete(rmse_job, outlier_run_ids, axis=0)

            n_runs = n_runs - len(outlier_run_ids)

        # Get the mean results now
        rmse_mean = np.mean(rmse_job, axis=0)
        rmse_std = np.std(rmse_job, axis=0)

        return rmse_job, n_runs, rmse_mean, rmse_std

    def plot_rmse(self, factor_filter=1.9, ind_iter_filter=[],
                  FLAG_plot=False, label_learners=['Uniform'], save_filename='results.png'):
        """
        Plots the RMSE (as recorded in the log-files) after removing outliers
        """
        rmse_job = self.job_results['rmse']
        n_runs = self.job_results['n_runs']

        if self.FLAG_filter is False:
            ind_iter_filter = []

        rmse_job, n_runs, rmse_mean, rmse_std = self.filter_outliers(rmse_job, n_runs,
                                                                     factor_filter=factor_filter,
                                                                     ind_iter_filter=ind_iter_filter)

        self.job_results['rmse'] = rmse_job
        self.job_results['n_runs'] = n_runs

        if FLAG_plot:
            plotting_options = {'save_filename': save_filename,
                                'label_learners': label_learners,
                                'FLAG_long_time_range': self.FLAG_long_time_range,
                                'FLAG_testing_error': False}
            plot_trend([rmse_mean], [self.N_p], error_learners=[rmse_std], plotting_options=plotting_options)

        return rmse_mean, rmse_std

    def plot_rmse_test(self, factor_filter=1.9, ind_iter_filter=[],
                       FLAG_plot=False, label_learners=['Uniform'], save_filename='results.png'):
        """
        Plots the RMSE (as recorded in the log-files) after removing outliers
        """
        rmse_test_job = self.job_results['rmse_test']
        n_runs_test = self.job_results['n_runs_test']

        if self.FLAG_filter is False:
            ind_iter_filter = []

        rmse_job, n_runs, rmse_mean, rmse_std = self.filter_outliers(rmse_test_job, n_runs_test,
                                                                     factor_filter=factor_filter,
                                                                     ind_iter_filter=ind_iter_filter)

        self.job_results['rmse_test'] = rmse_test_job
        self.job_results['n_runs_test'] = n_runs_test

        if FLAG_plot:
            plotting_options = {'save_filename': save_filename,
                                'label_learners': label_learners,
                                'FLAG_long_time_range': self.FLAG_long_time_range,
                                'FLAG_testing_error': False}
            plot_trend([rmse_mean], [self.N_p], error_learners=[rmse_std], plotting_options=plotting_options)

        return rmse_mean, rmse_std

    def plot_empirical_rmse(self, ind_iter_filter=[], factor_filter=1.9, n_iters_end=None,
                            FLAG_plot=False, label_learners=['Uniform'], save_filename='results.png'):
        """
        Compute and plot empirical rmse over the estimates from the log-files
        """
        max_iters = self.job_results['max_iters']
        n_runs = self.job_results['n_runs']

        if self.FLAG_filter is False:
            ind_iter_filter = []

        # Get empirical rmse
        rmse_iter_vec = np.zeros(shape=(n_runs, max_iters))
        rmse_trend = np.zeros(max_iters)
        J_mean = np.zeros(shape=(max_iters, 6))

        for ind_run in range(n_runs):
            J_mean += self.job_results['Jhat'][ind_run, :, :]

        J_mean = J_mean / n_runs

        for ind_run in range(n_runs):
            J_temp_run = self.job_results['Jhat'][ind_run, :, :]
            for ind_iter in range(max_iters):
                rmse_trend[ind_iter] += normalized_L2_error(J_temp_run[ind_iter, :],
                                                            J_mean[ind_iter, :], self.xi_J) ** 2

                rmse_iter_vec[ind_run, ind_iter] = normalized_L2_error(J_temp_run[ind_iter, :],
                                                                       J_mean[ind_iter, :], self.xi_J)

        _, _, rmse_trend, uncertainty_trend = self.filter_outliers(rmse_iter_vec, n_runs,
                                                                   factor_filter=factor_filter,
                                                                   ind_iter_filter=ind_iter_filter)
        if FLAG_plot:
            plotting_options = {'save_filename': save_filename,
                                'label_learners': label_learners,
                                'FLAG_long_time_range': self.FLAG_long_time_range,
                                'FLAG_testing_error': False,
                                'n_iters_end': n_iters_end}
            plot_trend([rmse_trend], [self.N_p], error_learners=[uncertainty_trend], plotting_options=plotting_options)

        return rmse_trend, uncertainty_trend, J_mean

    def compute_loss(self, X_test, FLAG_verbose=True):
        """
        Computes the log-likelihood error considering the testing dataset X_test
        :param X_test:
        :param FLAG_verbose:
        :return:
        """
        max_iters = self.job_results['max_iters']
        n_runs = self.job_results['n_runs']

        # Get empirical testing error
        test_iter_vec = np.zeros(shape=(n_runs, max_iters))

        # Fetch the MLE-Estimator to compute loss function
        mle_est = MLE_Estimator(X_test, self.xi_J)

        # For each run and iteration
        for ind_run in range(n_runs):
            J_temp_run = self.job_results['Jhat'][ind_run, :, :]
            for ind_iter in range(max_iters):
                J_temp_iter_run = J_temp_run[ind_iter, :]
                test_iter_vec[ind_run, ind_iter] = mle_est.np_loss(J_temp_iter_run, type_param='J')

            if FLAG_verbose and np.mod(ind_run, 5) == 0:
                print("Done with run %d" % ind_run)

        # Normalizing testing errors
        self.job_results.update({'loss_test': test_iter_vec})

    def plot_testing_error(self, X_test, J_truth, factor_filter=1.9, ind_iter_filter=[],
                           FLAG_recompute=False, FLAG_empirical=True,
                           FLAG_plot=False, FLAG_verbose=True,
                           label_learners=['Uniform'], save_filename='results.png'):
        """
        Computes and plots the testing error over the different estimates against a testing dataset

        Inputs:
            X_test: Testing dataset
            J_truth: Ground truth compared to which we will compute errors. It may also be the estimate from X_test
            FLAG_empirical: Use mean of theta's rather than J_truth
            FLAG_plot:
            label_learners:
            save_filename:
        """
        if FLAG_recompute:
            self.compute_loss(X_test, FLAG_verbose=FLAG_verbose)

        # Fetch loss function over testing dataset
        loss_test_job = np.copy(self.job_results['loss_test'])

        # Get empirical testing error
        n_runs = self.job_results['n_runs']
        max_iters = self.max_iters

        # Fetch the MLE-Estimator to compute loss function as required
        mle_est = MLE_Estimator(X_test, self.xi_J)

        if FLAG_empirical:
            print('Computing Empirical log-likelihood ratio')
            # Use J_mean
            _, _, J_mean = self.plot_empirical_rmse(FLAG_plot=False)

            loss_mean = np.zeros(max_iters)
            for ind_iter in range(max_iters):
                loss_mean[ind_iter] = mle_est.np_loss(J_mean[ind_iter, :], type_param='J')

                # Compute testing error
                loss_test_job[:, ind_iter] = loss_test_job[:, ind_iter] - loss_mean[ind_iter]
        else:
            print('Computing Ground log-likelihood ratio')
            # Use truth
            loss_truth = mle_est.np_loss(J_truth, type_param='J')

            # Compute testing error
            loss_test_job = loss_test_job - loss_truth

        if self.FLAG_filter is False:
            ind_iter_filter = []

        _, _, testing_error_mean, testing_error_std = self.filter_outliers(loss_test_job, n_runs,
                                                                           factor_filter=factor_filter,
                                                                           ind_iter_filter=ind_iter_filter)

        self.job_results.update({'testing_error_mean': testing_error_mean, 'testing_error_std': testing_error_std})

        if FLAG_plot:
            plotting_options = {'save_filename': save_filename,
                                'label_learners': label_learners,
                                'FLAG_long_time_range': self.FLAG_long_time_range,
                                'FLAG_testing_error': True, 'FLAG_fit_slopes': [True, True]}

            plot_trend([testing_error_mean], [self.N_p], error_learners=[testing_error_std],
                       plotting_options=plotting_options)

        return testing_error_mean, testing_error_std

    def plot_rmse_truth(self, J_truth, FLAG_plot=False,
                        label_learners=['Uniform'], save_filename='results.png'):
        """
        Compute and plot RMSE using estimates from log-files considering J_truth now as the ground-truth
        """
        max_iters = self.job_results['max_iters']
        n_runs = self.job_results['n_runs']

        # Get empirical rmse
        rmse_iter_vec = np.zeros(shape=(max_iters, n_runs))
        rmse_trend = np.zeros(max_iters)

        for ind_run in range(n_runs):
            J_temp_run = self.job_results['Jhat'][ind_run, :, :]
            for ind_iter in range(max_iters):
                rmse_trend[ind_iter] += normalized_L2_error(J_temp_run[ind_iter, :],
                                                            J_truth, self.xi_J) ** 2

                rmse_iter_vec[ind_iter, ind_run] = normalized_L2_error(J_temp_run[ind_iter, :],
                                                                       J_truth, self.xi_J)

        rmse_trend = np.sqrt(rmse_trend / n_runs)
        uncertainty_trend = np.std(rmse_iter_vec, axis=1)

        if FLAG_plot:
            plotting_options = {'save_filename': save_filename,
                                'label_learners': label_learners,
                                'FLAG_long_time_range': self.FLAG_long_time_range,
                                'FLAG_testing_error': False}

            plot_trend([rmse_trend], [self.N_p], error_learners=[uncertainty_trend], plotting_options=plotting_options)

        return rmse_trend, uncertainty_trend

    def plot_rmse_frequencies_truth(self, omega_truth, xi_theta=np.array([1e6, 1e6]), FLAG_plot=False,
                                    label_learners=['Uniform'], save_filename='results.png'):
        """
        Computes and plots the RMSE in frequencies
        """
        if self.FLAG_param_convert is not True:
            raise ValueError("Wasnt initialized properly! Check FLAG_param_convert!")

        n_runs = self.job_results['n_runs']
        max_iters = self.max_iters

        # Get empirical rmse for only the frequencies
        rmse_omega_iter_vec = np.zeros(shape=(max_iters, n_runs))
        rmse_omega_trend = np.zeros(max_iters)

        # Get the RMSE wrt omega_truth
        for ind_run in range(n_runs):
            param_temp_run = self.job_results['param_hat'][ind_run, :, :]
            for ind_iter in range(max_iters):
                omega_temp_run_iter = param_temp_run[ind_iter, [0,3]]
                rmse_omega_trend[ind_iter] += normalized_L2_error(omega_temp_run_iter,
                                                                  omega_truth, xi_theta) ** 2

                rmse_omega_iter_vec[ind_iter, ind_run] = normalized_L2_error(omega_temp_run_iter,
                                                                             omega_truth, xi_theta)

        rmse_omega_trend = np.sqrt(rmse_omega_trend / n_runs)
        error_omega_trend = np.std(rmse_omega_iter_vec, axis=1)

        if FLAG_plot:
            plotting_options = {'save_filename': save_filename,
                                'label_learners': label_learners,
                                'FLAG_long_time_range': self.FLAG_long_time_range,
                                'FLAG_testing_error': False}

            plot_trend([rmse_omega_trend], [self.N_p], error_learners=[error_omega_trend],
                       plotting_options=plotting_options)

        return rmse_omega_trend, error_omega_trend

    def plot_empirical_rmse_frequencies(self, xi_theta=np.array([1e6, 1e6]), FLAG_plot=False,
                                        label_learners=['Uniform'], save_filename='results.png'):
        """
        Computes and plots the RMSE in frequencies
        """
        if self.FLAG_param_convert is not True:
            raise ValueError("Wasnt initialized properly! Check FLAG_param_convert!")

        n_runs = self.job_results['n_runs']
        max_iters = self.max_iters

        # Get empirical rmse for only the frequencies
        param_mean = np.zeros(shape=(max_iters, 6))

        rmse_omega_iter_vec = np.zeros(shape=(max_iters, n_runs))
        rmse_omega_trend = np.zeros(max_iters)

        # Get the mean omega
        for ind_run in range(n_runs):
            param_mean += self.job_results['param_hat'][ind_run, :, :]

        param_mean = param_mean / n_runs
        omega_mean = param_mean[:, [0,3]]

        for ind_run in range(n_runs):
            param_temp_run = self.job_results['param_hat'][ind_run, :, :]
            for ind_iter in range(max_iters):
                omega_temp_run_iter = param_temp_run[ind_iter, [0,3]]
                rmse_omega_trend[ind_iter] += normalized_L2_error(omega_temp_run_iter,
                                                                  omega_mean[ind_iter, :], xi_theta) ** 2

                rmse_omega_iter_vec[ind_iter, ind_run] = normalized_L2_error(omega_temp_run_iter,
                                                                             omega_mean[ind_iter, :], xi_theta)

        rmse_omega_trend = np.sqrt(rmse_omega_trend / n_runs)
        error_omega_trend = np.std(rmse_omega_iter_vec, axis=1)

        if FLAG_plot:
            plotting_options = {'save_filename': save_filename,
                                'label_learners': label_learners,
                                'FLAG_long_time_range': self.FLAG_long_time_range,
                                'FLAG_testing_error': False}

            plot_trend([rmse_omega_trend], [self.N_p], error_learners=[error_omega_trend],
                       plotting_options=plotting_options)

        return rmse_omega_trend, error_omega_trend


def empirical_estimator_log(SAVE_DIR, max_iters=120, N_0=972, N_batch=100, FLAG_mod=True, J_truth=None,
                            xi_J=np.array([1e6 for _ in range(6)]),
                            FLAG_filter=True, factor_filter=1.9, ind_iter_filter=[0], FLAG_rmse_test=False,
                            do_plot=True, save_filename='results.png',
                            label_learners=['Uniform']):
    """
    Inputs:
        SAVE_DIR: Location of directory where all the log files of individual runs are saved
        max_iters: Maximum rounds/iterations of AL that we will consider (Error if it's higher than the min across logs)
        N_0, N_batch: As usual
        FLAG_mod: Compute empirical RMSE over estimates rather than using J_truth
        J_truth, xi_J: As usual
        FLAG_filter, factor_filter, ind_iter_filter: As usual for filtering out outliers from runs
        do_plot: FLAG for plotting
        FLAG_save: FLAG for saving plots made
        save_filename: name of file being saved
        label_learners: legend labels of learners being considered

    """
    emp_est = EmpiricalEstimator(SAVE_DIR, xi_J=xi_J, max_iters=max_iters, N_0=N_0, N_batch=N_batch,
                                 FLAG_filter=FLAG_filter, FLAG_rmse_test=FLAG_rmse_test, do_plot=do_plot)

    # Filter out the bad runs
    rmse_mean, rmse_std = emp_est.plot_rmse(factor_filter=factor_filter, ind_iter_filter=ind_iter_filter)

    # Load updated jobs
    rmse_job = emp_est.job_results['rmse']
    n_runs = emp_est.job_results['n_runs']

    rmse_errorbars = np.zeros((2, max_iters))
    for ind in range(max_iters):
        rmse_errorbars[0, ind] = np.abs(np.amin(rmse_job[:, ind]) - rmse_mean[ind])
        rmse_errorbars[1, ind] = np.abs(np.amax(rmse_job[:, ind]) - rmse_mean[ind])

    if FLAG_rmse_test:
        rmse_test_mean, rmse_test_std = emp_est.plot_rmse_test(factor_filter=factor_filter,
                                                               ind_iter_filter=ind_iter_filter)

    if FLAG_mod:
        # Get empirical rmse
        rmse_trend, uncertainty_trend, J_mean = emp_est.plot_empirical_rmse()

    if FLAG_mod:
        if do_plot:
            if FLAG_rmse_test:
                plot_trend([rmse_test_mean], [emp_est.N_p], error_learners=[rmse_test_std],
                           plotting_options={'label_learners': label_learners, 'save_filename': save_filename})
            else:
                plot_trend([rmse_trend], [emp_est.N_p], error_learners=[uncertainty_trend],
                           plotting_options={'label_learners': label_learners, 'save_filename': save_filename})

        results = {'rmse_mean': rmse_trend, 'rmse_error': uncertainty_trend, 'N_p': emp_est.N_p, 'Jhat_mean': J_mean}
    else:
        if do_plot:
            if FLAG_rmse_test:
                plot_trend([rmse_test_mean], [emp_est.N_p], error_learners=[rmse_test_std],
                           plotting_options={'label_learners': label_learners, 'save_filename': save_filename})
            else:
                plot_trend([rmse_mean], [emp_est.N_p], error_learners=[rmse_std],
                           plotting_options={'label_learners': label_learners, 'save_filename': save_filename})

        results = {'rmse_mean': rmse_mean, 'rmse_error': rmse_std, 'N_p': emp_est.N_p}

    if FLAG_rmse_test:
        results.update({'rmse_test_mean': rmse_test_mean})
        results.update({'rmse_test_std': rmse_test_std})

    return results


def scaling_learning_error(rmse_trend, nqueries_trend):
    return np.polyfit(np.log(nqueries_trend), np.log(rmse_trend), 1)[0]


def plot_trend_simulator_experimental_data(sim_runs, expt_runs, plotting_options=None):
    """
    Plotting function to compare simulator and experimental data with sim runs in the background and expt data in front

    Written primarily for short trends

    Inputs:
        sim_runs/expt_runs (dict) with keys of 'rmse', 'nqueries', 'error' in list format for the different learners

        Assuming the lists are ordered in the same fashion
    """
    # Extract inputs
    sim_rmse_learners = sim_runs['rmse_learners']
    sim_nqueries_learners = sim_runs['nqueries_learners']
    sim_error_learners = sim_runs['error_learners']
    sim_label_learerns = sim_runs['label_learners']

    expt_rmse_learners = expt_runs['rmse_learners']
    expt_nqueries_learners = expt_runs['nqueries_learners']
    expt_error_learners = expt_runs['error_learners']
    expt_label_learerns = expt_runs['label_learners']

    # Default plotting options
    default_plotting_options = {'FLAG_save_plot': True, 'save_filename': 'results.png',
                                'figsize_plot': (12, 10), 'skip_learner': 0,
                                'FLAG_fit_slopes': [True, True],
                                'n_iters_end': None, 'slope_scales': 0.9 * np.ones(len(sim_rmse_learners)),
                                'FLAG_reduced_HL': False,
                                'FLAG_testing_error': False, 'FLAG_long_time_range': False, 'FLAG_legend_outside': True}

    if plotting_options is None:
        plotting_options = default_plotting_options
    else:
        for _key in default_plotting_options.keys():
            plotting_options.setdefault(_key, default_plotting_options[_key])

    # Plot the trends -- Different from plot_trend as skipping plotting of passive learner
    color_lines = ['r-', 'g-', 'm-', 'r--', 'b--', 'g--', 'm--']
    marker_color = ['r', 'g', 'm', 'c', 'y']

    sim_marker_style = ['s', 's', 's', 's', 's', 's']
    expt_marker_style = ['x', 'x', 'x', 'x', 'x', 'x']

    # To force xlabels and ylabels to use the right defined Font
    # Ref: https://stackoverflow.com/questions/11611374/sans-serif-font-for-axes-tick-labels-with-latex
    from matplotlib import rc
    rc('text.latex', preamble=r'\usepackage{sfmath}')

    plt.figure(figsize=plotting_options['figsize_plot'])
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    # Let's mention log-scales early on
    plt.yscale('log')
    plt.xscale('log')

    # Where do we plot the slopes wrt data?
    slope_scales = plotting_options['slope_scales']

    # If we want to use the marker/color information starting from a different counter
    skip_learner = plotting_options['skip_learner']

    # Loop over the list of rmse for different learners (simulator)
    for ind in range(len(sim_rmse_learners)):
        # learner on sim
        sim_rmse_ind = sim_rmse_learners[ind]
        sim_nqueries_ind = sim_nqueries_learners[ind]
        sim_error_ind = sim_error_learners[ind]

        # learner on expt
        expt_rmse_ind = expt_rmse_learners[ind]
        expt_nqueries_ind = expt_nqueries_learners[ind]
        expt_error_ind = expt_error_learners[ind]

        # Plotting styles, labels, etc.
        color_learner_plot = marker_color[ind + skip_learner]
        marker_style_plot = sim_marker_style[ind + skip_learner]

        sim_label_learner_ind = sim_label_learerns[ind]
        expt_label_learner_ind = expt_label_learerns[ind]

        n_iters_end = plotting_options['n_iters_end']
        if n_iters_end is None:
            n_iters_end = round(len(sim_nqueries_ind) / 3)

        # PLOT SIMULATOR RESULTS
        if sim_error_ind.shape[0] == 2:
            plt.fill_between(sim_nqueries_ind, sim_rmse_ind - sim_error_ind[0, :], sim_rmse_ind + sim_error_ind[1, :],
                             color=color_learner_plot, alpha=0.2)
        else:
            plt.fill_between(sim_nqueries_ind, sim_rmse_ind - sim_error_ind, sim_rmse_ind + sim_error_ind,
                             color=color_learner_plot, alpha=0.2)

        plt.plot(sim_nqueries_ind, sim_rmse_ind, marker_style_plot,
                 color=color_learner_plot, mew=1.5, fillstyle='none', label=sim_label_learner_ind)

        # PLOT EXPT DATA RESULTS
        color_learner_plot = marker_color[ind + skip_learner]
        marker_style_plot = expt_marker_style[ind + skip_learner]

        # plt.plot(expt_nqueries_ind, expt_rmse_ind, marker_style_plot, color=color_learner_plot, fillstyle='none')

        # expt data plot with errorbars
        plt.errorbar(expt_nqueries_ind, expt_rmse_ind, yerr=expt_error_ind,
                     fmt=marker_style_plot, mew=1.5, color=color_learner_plot, capsize=5, capthick=2,
                     label=expt_label_learner_ind)

        # plt.vlines(expt_nqueries_ind, expt_rmse_ind + expt_error_ind, expt_rmse_ind-expt_error_ind,
        #            color=color_learner_plot)

        # RMSE of Learners (Fits) -- beginning and ending of trends
        if plotting_options['FLAG_fit_slopes'][0]:
            poly_init, cov_init = np.polyfit(np.log(sim_nqueries_ind[0:6]), np.log(sim_rmse_ind[0:6]), 1, cov=True)

            label_fit_init = r"slope = %0.2f $\pm$ %0.2f" % (poly_init[0], np.sqrt(np.diag(cov_init)[0]))

            plt.plot(sim_nqueries_ind[0:6], np.exp(np.polyval(poly_init, np.log(sim_nqueries_ind[0:6]))),
                     color=color_learner_plot, linestyle='dashed', label = label_fit_init)

        if plotting_options['FLAG_fit_slopes'][1]:
            poly_end, cov_end = np.polyfit(np.log(sim_nqueries_ind[-n_iters_end:]),
                                           np.log(sim_rmse_ind[-n_iters_end:]), 1, cov = True)

            label_fit_end = r"slope = %0.2f $\pm$ %0.2f" % (poly_end[0], np.sqrt(np.diag(cov_end)[0]))

            plt.plot(sim_nqueries_ind[-n_iters_end:],
                     slope_scales[ind] * np.exp(np.polyval(poly_end, np.log(sim_nqueries_ind[-n_iters_end:]))),
                     color=color_learner_plot, linestyle='-', label=label_fit_end)

    plt.xlabel("Number of queries")
    plt.ylabel("RMSE")

    plt.ylim([1.0e-4, 2])
    plt.ylabel("RMSE")

    plt.xticks([2e3, 5e3, 1e4, 2e4])
    plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$'])

    if plotting_options['FLAG_legend_outside']:
        plt.legend(bbox_to_anchor=(1.04, 1))
    else:
        # plt.legend(loc="upper right")
        plt.legend(loc='best')

    if plotting_options['FLAG_save_plot']:
        plt.savefig(plotting_options['save_filename'], bbox_inches='tight', dpi=600)


def plot_trend_compare_mle_bayesian(rmse_learners, nqueries_learners, nqueries_lower_bound=None,
                                    rmse_lower_bound=None, error_learners=None,
                                    FLAG_plot_error_learners=None,
                                    plotting_options=None):

    # Default plotting options
    default_plotting_options = {'label_learners': ['Uniform (Random)', 'AL FIR (Test: p_u)', 'AL FI (Test: Unknown)'],
                                'FLAG_save_plot': True, 'save_filename': 'results.png',
                                'figsize_plot': (12, 10), 'skip_learner': 0,
                                'FLAG_fit_slopes': [True, True],
                                'n_iters_end': None, 'slope_scales': 0.9*np.ones(len(rmse_learners)),
                                'FLAG_reduced_HL': False,
                                'FLAG_testing_error': False, 'FLAG_long_time_range': False, 'FLAG_legend_outside': True}

    if plotting_options is None:
        plotting_options = default_plotting_options
    else:
        for _key in default_plotting_options.keys():
            plotting_options.setdefault(_key, default_plotting_options[_key])

    # Plot the trends
    color_lines = ['r-', 'b-', 'g-', 'm-', 'r--', 'b--', 'g--', 'm--']
    marker_color = ['orange', 'hotpink', 'mediumturquoise', 'r', 'b', 'g']
    if plotting_options['FLAG_testing_error']:
        marker_color = ['orange', 'hotpink', 'r', 'b', 'g']

    #marker_style = ['ro', 'bo', 'go', 'mo', 'co', 'yo']
    marker_style = ['o', 'o', 'o', 'o', 'o', 'o']

    plt.figure(figsize=plotting_options['figsize_plot'])
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['text.usetex'] = True

    # change the scaling if testing error plots
    if plotting_options['FLAG_testing_error']:
        factor_rescale_plot = 100
    else:
        factor_rescale_plot = 1

    slope_scales = plotting_options['slope_scales']

    # If we want to use the marker/color information starting from a different counter
    skip_learner = plotting_options['skip_learner']

    # Loop over the list of rmse for different learners
    for ind in range(len(rmse_learners)):
        rmse_ind = factor_rescale_plot*rmse_learners[ind]
        nqueries_ind = nqueries_learners[ind]

        # Plotting styles, labels, etc.
        color_learner_plot = marker_color[ind + skip_learner]
        marker_style_plot = marker_style[ind + skip_learner]
        label_learner = plotting_options['label_learners'][ind]

        if FLAG_plot_error_learners is None:
            FLAG_plot_error = True
        else:
            FLAG_plot_error = FLAG_plot_error_learners[ind + skip_learner]

        n_iters_end = plotting_options['n_iters_end']
        if n_iters_end is None:
            n_iters_end = round(len(nqueries_ind) / 3)

        # RMSE of Learners (Data) with RMSE Errorbars if available
        if error_learners is not None:
            error_ind = factor_rescale_plot*error_learners[ind]

            if FLAG_plot_error:
                if error_ind.shape[0] == 2:
                    plt.fill_between(nqueries_ind, rmse_ind-error_ind[0,:], rmse_ind+error_ind[1,:],
                                     color=color_learner_plot, alpha=0.2)
                else:
                    plt.fill_between(nqueries_ind, rmse_ind - error_ind, rmse_ind + error_ind,
                                     color=color_learner_plot, alpha=0.2)

            # Errorbar might not be working
            # plt.errorbar(nqueries_ind, rmse_ind, yerr=error_ind,
            #              fmt=marker_style[ind], mec=marker_color[ind], uplims=True, lolims=True)

        if FLAG_plot_error:
            plt.plot(nqueries_ind, rmse_ind, marker_style_plot, color=color_learner_plot, label=label_learner)
        else:
            plt.plot(nqueries_ind, rmse_ind, color_learner_plot,
                     linestyle='dashed', linewidth=3, label=label_learner)

        # RMSE of Learners (Fits) -- beginning and ending of trends
        if FLAG_plot_error:
            if plotting_options['FLAG_fit_slopes'][0]:
                poly_init, cov_init = np.polyfit(np.log(nqueries_ind[0:6]), np.log(rmse_ind[0:6]), 1, cov=True)
                label_fit_init = r"slope = %0.2f $\pm$ %0.2f" % (poly_init[0], np.sqrt(np.diag(cov_init)[0]))

                plt.plot(nqueries_ind[0:6], np.exp(np.polyval(poly_init,
                                                              np.log(nqueries_ind[0:6]))),
                         color=color_learner_plot, linestyle='dashed', linewidth=2, label=label_fit_init)

            if plotting_options['FLAG_fit_slopes'][1]:
                poly_end, cov_end = np.polyfit(np.log(nqueries_ind[-n_iters_end:]),
                                               np.log(rmse_ind[-n_iters_end:]), 1, cov=True)

                label_fit_end = r"slope = %0.2f $\pm$ %0.2f" % (poly_end[0], np.sqrt(np.diag(cov_end)[0]))

                plt.plot(nqueries_ind[-n_iters_end:],
                         slope_scales[ind]*np.exp(np.polyval(poly_end, np.log(nqueries_ind[-n_iters_end:]))),
                         color=color_learner_plot, linestyle='-', linewidth=2.5, label=label_fit_end)

            # poly_all, cov_all = np.polyfit(np.log(nqueries_ind), np.log(rmse_ind), 1, cov=True)
            # label_fit_all = r"Slope = %0.2f $\pm$ %0.2f" % (poly_all[0], np.sqrt(np.diag(cov_all)[0]))
            # plt.plot(nqueries_ind, np.exp(np.polyval(poly_all, np.log(nqueries_ind))),
            #          'k:', label=label_fit_all)

    if rmse_lower_bound is not None and nqueries_lower_bound is not None:
        plt.plot(nqueries_lower_bound, rmse_lower_bound, 'k-', label='Cramer-Rao Bound')
        poly_crb, cov_crb = np.polyfit(np.log(nqueries_lower_bound), np.log(rmse_lower_bound), 1, cov=True)
        print(poly_crb)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Number of queries")

    if not plotting_options['FLAG_reduced_HL']:
        if plotting_options['FLAG_testing_error']:
            plt.ylabel(r"Testing Error ($\times 10^{-2}$)")

            if plotting_options['FLAG_long_time_range']:
                # Long time-range (FIR)
                plt.xticks([7e3, 1e4, 2e4, 5e4, 1e5])
                plt.gca().set_xticklabels([r'$7 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$', r'$5 \times 10^{4}$',
                                           r'$10^{5}$'])

                # plt.yticks([1.0e-3, 5.0e-4, 2.0e-4, 1.0e-4, 5.0e-5, 1.0e-5])
                # plt.gca().set_yticklabels([r'$10^{-3}$', r'$5 \times 10^{-4}$', r'$2 \times 10^{-4}$',
                #                            r'$10^{-4}$', r'$5 \times 10^{-4}$', r'$10^{-5}$'])
            else:
                # Short time-range (FI)
                plt.xticks([2e3,5e3,1e4,2e4,5e4,1e5])
                plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$',
                                           r'$5 \times 10^{4}$', r'$10^{5}$'])

                # plt.yticks([2e-2,5e-2,1e-1,2e-1,5e-1])
                # plt.gca().set_yticklabels([r'$2 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$',
                #                            r'$2 \times 10^{-1}$', r'$5 \times 10^{-1}$'])
        else:
            plt.ylabel("RMSE")
            if plotting_options['FLAG_long_time_range']:
                # Long time-range (FI)

                # Limits
                plt.ylim([2e-2, 0.6])

                plt.xticks([7e3,1e4,2e4,5e4,1e5])
                plt.gca().set_xticklabels([r'$7 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$', r'$5 \times 10^{4}$',
                                           r'$10^{5}$'])

                plt.yticks([2e-2,5e-2,1e-1,2e-1,5e-1])
                plt.gca().set_yticklabels([r'$2 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$',
                                           r'$2 \times 10^{-1}$', r'$5 \times 10^{-1}$'])
            else:
                # Short time-range (FI)

                # Limits
                plt.ylim([1e-2, 2.5])

                # Tick-marks
                plt.xticks([2e3,5e3,1e4,2e4,5e4,1e5])
                plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$',
                                           r'$5 \times 10^{4}$', r'$10^{5}$'])

                plt.yticks([2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0])
                plt.gca().set_yticklabels([r'$2 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$',
                                           r'$2 \times 10^{-1}$', r'$5 \times 10^{-1}$', r'$10^{0}$', r'$2 \times 10^{0}$'])
    else:
        plt.ylim([1.0e-4, 2])
        plt.ylabel("RMSE")
        if plotting_options['FLAG_long_time_range']:
            plt.xticks([2e3, 5e3, 1e4, 2e4, 5e4])
            plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$',
                                       r'$5 \times 10^{4}$'])
        else:
            plt.xticks([2e3, 5e3, 1e4, 2e4])
            plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$'])

    if plotting_options['FLAG_legend_outside']:
        plt.legend(bbox_to_anchor=(1.04, 1))
        # plt.legend(bbox_to_anchor=(0., -0.5, 1., 0.02),
        #            loc='lower center', ncol=2)
    else:
        #plt.legend(loc="upper right")
        # plt.legend(loc='best')
        plt.legend(loc='best', ncol=2)

    if plotting_options['FLAG_save_plot']:
        plt.savefig(plotting_options['save_filename'], bbox_inches='tight', dpi=300)