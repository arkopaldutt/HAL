"""
Function for PL experiments on simulator with baseline solver considering Nyquist criterion is satisfied
"""
import os, sys, shutil
import pathlib
import argparse
import numpy as np
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore")

# Add the package hamiltonianlearner
PROJECT_PATH = str(pathlib.Path().resolve().parent)
sys.path.append(PROJECT_PATH)

import hamiltonianlearner.quantum_system_oracles.simulate_nature as simulate_nature
import hamiltonianlearner.quantum_system_models.quantum_device_models as quantum_device_models
import hamiltonianlearner.learners.design_experiment as design_experiment
import hamiltonianlearner.utils.job_helper as job_helper


##------------------------------------------##
##                  MAIN                    ##
##------------------------------------------##

# 1. Setup a simulator (oracle) based on the experimental data from a quantum device

# Device (ibmq_boeblingen) to be simulated -- Load the data from which we will consider the parameters and time-stamps
ibm_data = job_helper.setup_oracle()

print('Using J_num which we already discovered before!')
J_truth = np.array([-4568258.88132824, -1465045.02848701,  -290468.97835928,
                    6499545.9801579,  1390900.08768167,   413722.74870734])

param_truth = quantum_device_models.transform_parameters(J_truth)

# Oracle properties
FLAG_simulator = True

# Noise Models
FLAG_readout_noise = False
FLAG_control_noise = True

# Control Noise
teff = quantum_device_models.data_driven_teff_noise_model(param_truth, FLAG_ibmq_boel=True)
expt_noise ={'readout': ibm_data['misclassif_error'], 'imperfect_pulse_shaping': teff}

# Create oracle
oracle_qs_sim = simulate_nature.Nature(J_truth, noise=expt_noise, expt_data=None,
                                       FLAG_simulator=FLAG_simulator,
                                       FLAG_readout_noise=FLAG_readout_noise,
                                       FLAG_control_noise=FLAG_control_noise)

# 2. Query space and ActionSpace setup
time_stamps = np.linspace(1.0e-7, 18e-7, 3*81)
query_space, A_cr = job_helper.setup_action_space(time_stamps, J_truth)

# Illustration of different freq_converts
print("freq_convert from sim: %f" % A_cr.freq_convert)
print("freq_convert from expt data: %f" % ibm_data['freq_convert'])

# Setup the ActiveLearner
policy_type = 'FI'
FLAG_normalization = True
FLAG_noise = True
growth_time_stamps = None
FLAG_constraints = True

query_constraints_ref = {'N_shots': 512}
FLAG_lower_limits = False

HAL_FI_J = design_experiment.ActiveLearner(policy_type=policy_type,
                                           FLAG_normalization=FLAG_normalization,
                                           FLAG_noise=FLAG_noise,
                                           FLAG_constraints=FLAG_constraints,
                                           query_constraints=query_constraints_ref,
                                           type_param='J')

####!!!! Need to add query constraints properly to the above procedure!!!

# 3. Define estimator
est_param_info = {'freq_convert': A_cr.freq_convert, 'xi_t': A_cr.xi_t, 'xi_J': A_cr.xi_J, 'init_J': None}
estimation_strategy = {'baseline': False,
                       'quick_MLE': True,
                       'FLAG_initial_estimate': True,
                       'FLAG_FFT_high_resolution': False,
                       'FLAG_MLE_param': False,
                       'FLAG_MLE_J': False}

# 4. Define conditions for experiment runner
max_iter = 70
N_0 = 5*A_cr.N_actions
N_batch = A_cr.N_actions

FLAG_query_constraints = True
query_constraints_info = {'query_constraints_ref': None, 'query_optimization_type': 'batch', 'max_iter': max_iter}
FLAG_adaptive_query_space = False
adaptive_query_space_info = {'growth_time_stamps': 'linear', 'max_iter_growth': 1, 'growth_factor': 1}

# 5. For saving and logging info
# Creation of save directory
expt_number = 0
FLAG_job_restart = False
FLAG_runs_restart = False
FLAG_debug_mode = False

if FLAG_simulator:
    SAVE_DIR = policy_type + '_mle_sim_long_%03d' % expt_number
else:
    SAVE_DIR = policy_type + '_mle_expt_long_%03d' % expt_number

if FLAG_debug_mode:
    SAVE_DIR = SAVE_DIR + '_debug'

# Create log-file (summary of all runs) and denote entries
log_filename = SAVE_DIR + '/log_job_%d.txt' % expt_number

if not os.access(SAVE_DIR, os.F_OK):
    os.makedirs(SAVE_DIR)

    if not FLAG_job_restart:
        f_log = open(log_filename, "a+")
        f_log.write("Run Iter RMSE Test_Error\n")
        f_log.close()

# Save script to folder to know what we ran
SAVE_DIR_script = SAVE_DIR + '/script'

if not os.access(SAVE_DIR_script, os.F_OK):
    os.makedirs(SAVE_DIR_script)

current_script_file = os.path.basename(__file__)
shutil.copy(current_script_file, SAVE_DIR_script)

# log file for a particular run
run_log_filename = SAVE_DIR + '/run_log_job_%d.txt' % expt_number

# For the runs
parser = argparse.ArgumentParser(description='PL Run')
parser.add_argument('--run_seed', type=int, default=0, metavar='N')
args = parser.parse_args()

# Define the experiment
AL_expt_run = partial(job_helper.active_learner_expt, max_iter=max_iter, env_qs=oracle_qs_sim,
                      query_space=query_space,
                      est_param_info=est_param_info,
                      active_learner=HAL_FI_J,
                      FLAG_query_constraints=FLAG_query_constraints,
                      query_constraints_info=query_constraints_info,
                      estimation_strategy=estimation_strategy,
                      FLAG_adaptive_query_space=FLAG_adaptive_query_space,
                      adaptive_query_space_info=adaptive_query_space_info,
                      N_0=N_0, N_batch=N_batch,
                      log_filename=log_filename,
                      run_log_filename=run_log_filename,
                      SAVE_DIR=SAVE_DIR, FLAG_debug_mode=FLAG_debug_mode)

print("Going to carry out run %d with %d iterations" %(args.run_seed, max_iter))

mse_U = np.zeros(max_iter + 1)

start_time = time.perf_counter()

if __name__ == '__main__':
    results_passive_learner = AL_expt_run(args.run_seed)


finish_time = time.perf_counter()

print(f'Finished in {round(finish_time - start_time, 2)} second(s)')