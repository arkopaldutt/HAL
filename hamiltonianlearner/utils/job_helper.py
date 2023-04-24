"""
Includes functions to help running job scripts
"""
import numpy as np
import pickle
import scipy.linalg

# package imports
from .learner_experiment_utils import Learning_Experiment_Runner
from .. import estimators
from ..quantum_system_oracles import process_data, simulate_nature


def setup_oracle(ind_amp=1, FLAG_classification=True):
    """
    Quick helper function to get an experimental dataset for setting up an oracle
    """
    # Parameters of the different jobs
    meas_level_expt = 1
    n_shots = 512
    n_job = 1
    cr_amp_array = [0.24, 0.30, 0.36, 0.42, 0.48]

    # Load data
    pickle_result_filename = 'ibmq_boel_fixed_qs_data_aligned_A_0_%d_meas_%d_shots_%d_job_%d.pickle' % (
        100 * cr_amp_array[ind_amp], meas_level_expt, n_shots, n_job)

    pickle_result_file = 'Data/ibmq_boel/' + pickle_result_filename

    # Readout calibration and formatting again
    ibm_data = process_data.make_dataset_ibmq_device(pickle_result_file,
                                                     FLAG_classification=FLAG_classification, do_plot=False)

    return ibm_data


def setup_oracle_amp(amp=0.3, FLAG_classification=True):
    """
    Quick helper function to get an experimental dataset for setting up an oracle from the adaptive query space datasets
    """
    # Parameters of the different jobs
    ind_job = 0

    # Load data
    amp = int(100*amp)

    pickle_result_filename = "ibmq_boel_adaptive_qs_data_aligned_A_0_%d_meas_1_shots_512_job_%d.pickle" % (amp, ind_job)

    pickle_result_file = 'Data/ibmq_boel/' + pickle_result_filename

    # Readout calibration and formatting again
    ibm_data = process_data.make_dataset_ibmq_device(pickle_result_file,
                                                     FLAG_classification=FLAG_classification, do_plot=False)

    return ibm_data


def setup_action_space(time_stamps, J_truth, n_shots=1e8):
    """
    Quick helper function to setup the ActionSpace given experimental data set on which the oracle is based

    Input:
        ibm_data: Dataset of interest
        J_truth: True Hamiltonian parameter values

    Returns:
        ActionSpace
    """
    # Normalizing factors
    xi_J = 10 ** np.amax(np.floor(np.log10(np.abs(J_truth)))) * np.ones(len(J_truth))
    xi_t = 1e-7

    ## Create Oracle
    si = np.array([[1, 0], [0, 1]])
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])

    moset = {0: [si, scipy.linalg.expm(1j * (np.pi / 4) * sy)],
             1: [si, scipy.linalg.expm(-1j * (np.pi / 4) * sx)], 2: [si, si]}
    prepset = {0: [si, si], 1: [sx, si]}

    time_stamps_nd = time_stamps/xi_t

    # Create the query space
    query_space = {'moset': moset, 'prepset': prepset, 'time_stamps': time_stamps}

    # Create the action space corresponding to simulator
    A_cr = simulate_nature.Action_Space(moset, prepset, time_stamps_nd, xi_t, xi_J=xi_J, n_shots=n_shots)

    return query_space, A_cr


def passive_learner_expt(i_run, max_iter, env_qs, query_space, est_param_info,
                        active_learner, FLAG_query_constraints, query_constraints_info,
                        estimation_strategy, FLAG_adaptive_query_space, adaptive_query_space_info,
                        N_0, N_batch, log_filename, run_log_filename, SAVE_DIR, FLAG_debug_mode):
    """
    Quick helper function to carry out an experiment using the input active learner
    Inputs:
        i_run: Seed for this run
        max_iter: Maximum number of iterations
        env_qs: Oracle/Environment of the quantum device
        query_space: Dictionary containing keys of moset, prepset and time_stamps
            moset: Set of measurement operators (expecting dict)
            prepset: Set of preparation operators (expecting dict)
            time_stamps: Set of time stamps (array)
        xi_t:
        xi_J:
        active_learner:
        FLAG_constraints:
        freq_convert:
        N_0:
        N_batch:
        log_filename:
        run_log_filename:
        SAVE_DIR:

    Returns:
    """
    np.random.seed(10 * (i_run + 2))
    print('Run %d with Passive Learning \n Policy Type: %s, Oracle: Simulator (%s)'
          % (i_run, active_learner.policy_type, env_qs.FLAG_simulator))

    if active_learner.policy_type != 'passive':
        print('This experiment is for PL. Indicated a different policy type. Running PL instead')

    log_file_run = SAVE_DIR + '/log_Run_%d.txt' % i_run

    # Setup experiment runner
    PL_expt = Learning_Experiment_Runner(env_cr=env_qs,
                                         query_space=query_space,
                                         est_param_info=est_param_info,
                                         active_learner=active_learner,
                                         FLAG_query_constraints=FLAG_query_constraints,
                                         query_constraints_info=query_constraints_info,
                                         estimation_strategy=estimation_strategy,
                                         FLAG_adaptive_query_space=FLAG_adaptive_query_space,
                                         adaptive_query_space_info=adaptive_query_space_info,
                                         N_0=N_0, N_batch=N_batch,
                                         FLAG_debug_AL=FLAG_debug_mode)

    results_FI = PL_expt.PL_runner(log_file=log_file_run)

    # Update RMSE and log results
    f_log = open(log_filename, "a+")

    loss_hat = results_FI['loss']
    mse = results_FI['mse']

    for i_iter in range(max_iter + 1):
        f_log.write("%d %d %f %f \n" % (i_run, i_iter, np.sqrt(mse[i_iter]), loss_hat[i_iter]))

    f_log.close()

    pickle_result_file = SAVE_DIR + '/Run_%d.pickle' % i_run

    with open(pickle_result_file, 'wb') as handle:
        pickle.dump(results_FI, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results_FI

    # except:
    #     print('Bracketing interval error most likely')
    #
    #     # Update the run log
    #     g_log = open(run_log_filename, "a+")
    #     g_log.write("Run %d Failed \n" % i_run)
    #     g_log.close()
    #
    #     return None


def active_learner_expt(i_run, max_iter, env_qs, query_space, est_param_info,
                        active_learner, FLAG_query_constraints, query_constraints_info,
                        estimation_strategy, FLAG_adaptive_query_space, adaptive_query_space_info,
                        N_0, N_batch, log_filename, run_log_filename, SAVE_DIR, FLAG_debug_mode):
    """
    Quick helper function to carry out an experiment using the input active learner
    Inputs:
        i_run: Seed for this run
        max_iter: Maximum number of iterations
        env_qs: Oracle/Environment of the quantum device
        query_space: Dictionary containing keys of moset, prepset and time_stamps
            moset: Set of measurement operators (expecting dict)
            prepset: Set of preparation operators (expecting dict)
            time_stamps: Set of time stamps (array)
        xi_t:
        xi_J:
        active_learner:
        FLAG_constraints:
        freq_convert:
        N_0:
        N_batch:
        log_filename:
        run_log_filename:
        SAVE_DIR:

    Returns:
    """
    np.random.seed(10 * (i_run + 2))
    print('Run %d with Active Learning \n Policy Type: %s, Oracle: Simulator (%s)'
          % (i_run, active_learner.policy_type, env_qs.FLAG_simulator))

    log_file_run = SAVE_DIR + '/log_Run_%d.txt' % i_run

    # Setup experiment runner
    AL_expt = Learning_Experiment_Runner(env_cr=env_qs,
                                         query_space=query_space,
                                         est_param_info=est_param_info,
                                         active_learner=active_learner,
                                         FLAG_query_constraints=FLAG_query_constraints,
                                         query_constraints_info=query_constraints_info,
                                         estimation_strategy=estimation_strategy,
                                         FLAG_adaptive_query_space=FLAG_adaptive_query_space,
                                         adaptive_query_space_info=adaptive_query_space_info,
                                         N_0=N_0, N_batch=N_batch,
                                         FLAG_debug_AL=FLAG_debug_mode)

    results_FI = AL_expt.AL_runner(log_file=log_file_run)

    # Update RMSE and log results
    f_log = open(log_filename, "a+")

    loss_hat = results_FI['loss']
    mse = results_FI['mse']

    for i_iter in range(max_iter + 1):
        f_log.write("%d %d %f %f \n" % (i_run, i_iter, np.sqrt(mse[i_iter]), loss_hat[i_iter]))

    f_log.close()

    pickle_result_file = SAVE_DIR + '/Run_%d.pickle' % i_run

    with open(pickle_result_file, 'wb') as handle:
        pickle.dump(results_FI, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results_FI

    # except:
    #     print('Bracketing interval error most likely')
    #
    #     # Update the run log
    #     g_log = open(run_log_filename, "a+")
    #     g_log.write("Run %d Failed \n" % i_run)
    #     g_log.close()
    #
    #     return None
