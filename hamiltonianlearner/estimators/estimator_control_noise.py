import numpy as np
import math
import cmath
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar, curve_fit

import matplotlib.pyplot as plt
from ..quantum_system_oracles import simulate_nature, process_data
from ..quantum_system_models import quantum_device_models


# Function for estimation parameters of noise due to Imperfect Pulse-Shaping
def get_first_order_teff_noise_model_old_ibm_devices(device_file, n_drive_configs, FLAG_verbose=True,
                                                     do_plot=True, plot_type='loglog', FLAG_save=False,
                                                     filename_plot='control_noise_model_dev.eps'):
    # Model fitters
    def curve_func(x, a, b):
        return a * (x ** b)

    def curve_func2(x, a, b):
        return a / (x + b * x ** 2)

    def curve_func3(x, a, b, c):
        return a / (x + b * x ** 2 + c * x ** 3)

    # Compute estimates
    omega_array_ibmq = []
    teff_array_ibmq = []

    for ind_amp in range(n_drive_configs):
        print(ind_amp)
        # load the device data of choice
        ibm_data = process_data.make_dataset_device(ind_amp, data_file_name=device_file,
                                                    FLAG_classification=True, do_plot=False,
                                                    FLAG_save=False)

        # Get initial estimates
        rabi_freq, rabi_gpt = initial_estimate(ibm_data, FLAG_syn_data=True)

        param_array_IC = np.array([rabi_freq[0] / 2, 2 * (rabi_gpt[0, 2] - np.pi / 4), rabi_gpt[0, 1],
                                   rabi_freq[1] / 2, 2 * (rabi_gpt[1, 2] - np.pi / 4), rabi_gpt[1, 1]])

        J_num = quantum_device_models.transform_theta_to_J(param_array_IC)

        teff0 = rabi_gpt[0, 0] / rabi_freq[0]
        teff1 = rabi_gpt[1, 0] / rabi_freq[1]

        omega0 = param_array_IC[0]
        omega1 = param_array_IC[3]

        # Convert all the teff to negative numbers -- mainly required for Device A for some reason
        if device_file == 'Data/device_A/hamiltonian_tomo_data.p':
            while teff0 > 0:
                teff0 -= 2*np.pi/omega0

            while teff1 > 0:
                teff1 -= 2*np.pi/omega1

        # Convert all the teff to positive numbers
        teff0 += 2 * np.pi / omega0
        teff1 += 2 * np.pi / omega1

        while teff0 < 0:
            teff0 += 2 * np.pi / omega0

        while teff1 < 0:
            teff1 += 2 * np.pi / omega1

        if FLAG_verbose:
            r0, r1 = ibm_data['misclassif_error']

            print('Readout noise: %0.3f, %0.3f' %(np.round(r0, 4), np.round(r1, 4)))
            print('Teff (ns): %d, %d' % (np.round(teff0/1.e-9, 0), np.round(teff1/1.e-9, 0)))
            print('omega0, omega1 [x1e6]: %.2f, %.2f' % (np.round(omega0/1e6, 2),
                                                         np.round(omega1/1e6, 2)))
            J_num_mod = np.round(J_num/1e6, 2)
            print('%.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (J_num_mod[0], J_num_mod[1], J_num_mod[2],
                                                          J_num_mod[3], J_num_mod[4], J_num_mod[5]))

        # append to arrays
        omega_array_ibmq.append(omega0)
        omega_array_ibmq.append(omega1)

        teff_array_ibmq.append(teff0)
        teff_array_ibmq.append(teff1)

    # Sort the arrays according to array of omegas - Estimated values
    omega_sorted_est = np.sort(omega_array_ibmq)
    sort_omega_index_est = np.argsort(omega_array_ibmq)
    teff_sorted_omega_index_est = np.array([teff_array_ibmq[ind] for ind in sort_omega_index_est])

    if device_file == 'Data/device_A/hamiltonian_tomo_data.p':
        if teff_sorted_omega_index_est[-1] < 2e-8:
            teff_sorted_omega_index_est[-1] += np.pi / omega_sorted_est[-1]

    # Fitter
    popt2, pcov2 = curve_fit(curve_func2, omega_sorted_est, teff_sorted_omega_index_est)

    # Error bars on curve fits
    #print(np.sqrt(np.diag(pcov2)))

    # Mean error between model and data
    print(np.mean((curve_func2(omega_sorted_est, *popt2) - teff_sorted_omega_index_est) ** 2))

    # Numerical model for omega/delta_t
    omega_fit = np.linspace(np.min(omega_sorted_est), np.max(omega_sorted_est), 100)
    delta_t_fit2 = popt2[0] / (omega_fit + popt2[1] * (omega_fit) ** 2)

    if do_plot:
        marker_color = ['b', 'g', 'r', 'c', 'm', 'y']
        Dt_nd = 10 ** (int(np.log10(np.max(teff_sorted_omega_index_est))))
        omega_nd = 10 ** (int(np.log10(np.max(omega_sorted_est))))

        plt.figure(6, figsize=(8, 8))
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams['font.sans-serif'] = "Helvetica"
        plt.rcParams['text.usetex'] = True
        plt.rcParams['figure.titlesize'] = 24
        plt.rcParams['axes.labelsize'] = 24
        plt.rcParams['axes.titlesize'] = 24
        plt.rcParams['legend.fontsize'] = 24
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        # Plot Data
        if plot_type == 'loglog':
            plt.loglog(omega_fit / omega_nd, delta_t_fit2 / Dt_nd, '-k', label='Fit ')
            plt.loglog(omega_sorted_est / omega_nd, teff_sorted_omega_index_est / Dt_nd, 'om')
        else:
            plt.plot(omega_fit / omega_nd, delta_t_fit2 / Dt_nd, '-k', label='Fit ')
            plt.plot(omega_sorted_est / omega_nd, teff_sorted_omega_index_est / Dt_nd, 'om')

        # plt.errorbar(omega_array_devD0_2/omega_nd, delta_t_devD0_2/Dt_nd, yerr=model_error_devD0_2/Dt_nd, fmt=marker_style[3], mec=marker_color[3], uplims=True, lolims=True, label='Dev D CR0_2')
        #a = str(int(np.log10(np.max(omega_sorted_est))))
        #b = str(int(np.log10(np.max(teff_sorted_omega_index_est))))

        xlabel_fig = r'$\omega$ ($10^7\,\mathrm{s}^{-1}$)'
        ylabel_fig = r'Time Offset ($10^{-5}$ s)'
        plt.xlabel(xlabel_fig, labelpad=10)
        plt.ylabel(ylabel_fig, labelpad=10)

        if plot_type == 'loglog':
            # Device D/CR0_1
            xticks_plot = [0.1, 0.2, 0.5, 1.0, 1.5, 3.0]
            plt.xticks(xticks_plot)
            plt.gca().set_xticklabels(['0.1', '0.2', '0.5', '1.0', '1.5', '3.0'])

            yticks_plot = [0.02, 0.05, 0.1, 0.2, 0.5]
            plt.yticks(yticks_plot)
            plt.gca().set_yticklabels(['0.02', '0.05', '0.1', '0.2', '0.5'])

            plt.text(0.44, 0.05, r'Curve Fit: $\Delta t = \frac{a}{\omega + b \omega^2}$',
                     {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center'})
            plt.text(0.44, 0.04, r'$a = 6.6537 \pm 0.05237$',
                     {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})
            plt.text(0.44, 0.03, r'$b = 1.1963 \times 10^{-8} \pm 0.2622 \times 10^{-8}$ (s)',
                     {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})

            # Device D/CR0_2
            # xticks_plot = [0.1, 0.2, 0.5, 1.0, 1.5, 3.0]
            # plt.xticks(xticks_plot)
            # plt.gca().set_xticklabels(['0.1', '0.2', '0.5', '1.0', '1.5', '3.0'])
            #
            # yticks_plot = [0.02, 0.05, 0.1, 0.2, 0.5]
            # plt.yticks(yticks_plot)
            # plt.gca().set_yticklabels(['0.02', '0.05', '0.1', '0.2', '0.5'])
            #
            # plt.text(0.44, 0.05, r'Curve Fit: $\Delta t = \frac{a}{\omega + b \omega^2}$',
            #          {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center'})
            # plt.text(0.44, 0.04, r'$a = 6.0991 \pm 0.03305$',
            #          {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})
            # plt.text(0.44, 0.03, r'$b = 3.2425 \times 10^{-9} \pm 0.8332 \times 10^{-9}$ (s)',
            #          {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})

            # # Device C
            # xticks_plot = [0.1, 0.2, 0.5, 1.0, 1.5, 3.0]
            # plt.xticks(xticks_plot)
            # plt.gca().set_xticklabels(['0.1', '0.2', '0.5', '1.0', '1.5', '3.0'])
            #
            # yticks_plot = [0.02,0.05,0.1,0.2,0.5]
            # plt.yticks(yticks_plot)
            # plt.gca().set_yticklabels(['0.02', '0.05', '0.1', '0.2', '0.5'])
            #
            # plt.text(0.44, 0.04, r'Curve Fit: $\Delta t = \frac{a}{\omega + b \omega^2}$',
            #          {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center'})
            # plt.text(0.44, 0.03, r'$a = 6.2812 \pm 0.01613$',
            #          {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})
            # plt.text(0.44, 0.02, r'$b = 9.1003 \times 10^{-9} \pm 0.7479 \times 10^{-9}$ (s)',
            #          {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})

            # # Device A
            # xticks_plot = [1, 2, 4, 6, 10]
            # plt.xticks(xticks_plot)
            # plt.gca().set_xticklabels(['1', '2', '4', '6', '10'])
            #
            # yticks_plot = [0.02,0.05,0.1,0.2,0.5]
            # plt.yticks(yticks_plot)
            # plt.gca().set_yticklabels(['0.02', '0.05', '0.1', '0.2', '0.5'])
            #
            # plt.text(2, 0.06, r'Curve Fit: $\Delta t = \frac{a}{\omega + b \omega^2}$',
            #          {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center'})
            # plt.text(2, 0.05, r'$a = 6.2419 \pm 0.10466$',
            #          {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})
            # plt.text(2, 0.04, r'$b = 5.8517 \times 10^{-9} \pm 1.4072 \times 10^{-9}$ (s)',
            #          {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})
        else:
            plt.text(1.0, 0.25, r'Curve Fit: $\Delta t = \frac{a}{\omega + b \omega^2}$',
                     {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center'})
            plt.text(1.0, 0.22, r'$a = 6.2774 \pm 0.01502$',
                     {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})
            plt.text(1.0, 0.19, r'$b = 1.5086 \times 10^{-9} \pm 0.6104 \times 10^{-9}$ (s)',
                     {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})

        plt.legend(loc='upper right')

        if FLAG_save:
            plt.savefig(filename_plot, bbox_inches='tight')

        plt.show()

    return popt2, np.sqrt(np.diag(pcov2))


def get_first_order_teff_noise_model(cr_amp_array, meas_level_expt, n_shots, n_job,
                                     do_plot=True, plot_type='loglog',
                                     filename_plot='control_noise_model_ibmq_boeb2_loglog.eps'):
    # Model fitters
    def curve_func(x, a, b):
        return a * (x ** b)

    def curve_func2(x, a, b):
        return a / (x + b * x ** 2)

    def curve_func3(x, a, b, c):
        return a / (x + b * x ** 2 + c * x ** 3)

    # Compute estimates
    omega_array_ibmq = []
    teff_array_ibmq = []

    for ind_amp in range(len(cr_amp_array)):
        # load the result file
        pickle_result_filename = 'ibmq_boel_fixed_qs_data_aligned_A_0_%d_meas_%d_shots_%d_job_%d.pickle' % (
        100 * cr_amp_array[ind_amp], meas_level_expt,
        n_shots, n_job)
        pickle_result_file = 'Data/ibmq_boel/' + pickle_result_filename

        # process the data into desired format
        ibm_data = process_data.make_dataset_ibmq_device(pickle_result_file, FLAG_classification=True, do_plot=False)

        # Get initial estimates
        rabi_freq, rabi_gpt = initial_estimate(ibm_data, FLAG_syn_data=True)

        param_array_IC = np.array([rabi_freq[0] / 2, 2 * (rabi_gpt[0, 2] - np.pi / 4), rabi_gpt[0, 1],
                                   rabi_freq[1] / 2, 2 * (rabi_gpt[1, 2] - np.pi / 4), rabi_gpt[1, 1]])

        J_num = quantum_device_models.transform_theta_to_J(param_array_IC)

        teff0 = rabi_gpt[0, 0] / rabi_freq[0]
        teff1 = rabi_gpt[1, 0] / rabi_freq[1]

        omega0 = param_array_IC[0]
        omega1 = param_array_IC[3]

        # Convert all the teff to positive numbers
        teff0 += 2 * np.pi / omega0
        teff1 += 2 * np.pi / omega1

        while teff0 < 0:
            teff0 += 2 * np.pi / omega0

        while teff1 < 0:
            teff1 += 2 * np.pi / omega1

        #     # Convert all the teff to negative numbers
        #     while teff0 > 0:
        #         teff0 -= 2*np.pi/omega0

        #     while teff1 > 0:
        #         teff1 -= 2*np.pi/omega1

        # append to arrays
        omega_array_ibmq.append(omega0)
        omega_array_ibmq.append(omega1)

        teff_array_ibmq.append(teff0)
        teff_array_ibmq.append(teff1)

    # Sort the arrays according to array of omegas - Estimated values
    omega_sorted_est = np.sort(omega_array_ibmq)
    sort_omega_index_est = np.argsort(omega_array_ibmq)
    teff_sorted_omega_index_est = np.array([teff_array_ibmq[ind] for ind in sort_omega_index_est])

    # Fitter
    popt2, pcov2 = curve_fit(curve_func2, omega_sorted_est, teff_sorted_omega_index_est)

    # Error bars on curve fits
    #print(np.sqrt(np.diag(pcov2)))

    # Mean error between model and data
    print(np.mean((curve_func2(omega_sorted_est, *popt2) - teff_sorted_omega_index_est) ** 2))

    # Numerical model for omega/delta_t
    omega_fit = np.linspace(np.min(omega_sorted_est), np.max(omega_sorted_est), 100)
    delta_t_fit2 = popt2[0] / (omega_fit + popt2[1] * (omega_fit) ** 2)

    if do_plot:
        marker_color = ['b', 'g', 'r', 'c', 'm', 'y']
        Dt_nd = 10 ** (int(np.log10(np.max(teff_sorted_omega_index_est))))
        omega_nd = 10 ** (int(np.log10(np.max(omega_sorted_est))))

        plt.figure(6, figsize=(8, 8))
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams['font.sans-serif'] = "Helvetica"
        plt.rcParams['text.usetex'] = True
        plt.rcParams['figure.titlesize'] = 24
        plt.rcParams['axes.labelsize'] = 24
        plt.rcParams['axes.titlesize'] = 24
        plt.rcParams['legend.fontsize'] = 24
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20
        # Plot Data
        if plot_type == 'loglog':
            plt.loglog(omega_fit / omega_nd, delta_t_fit2 / Dt_nd, '-k', label='Fit ')
            plt.loglog(omega_sorted_est / omega_nd, teff_sorted_omega_index_est / Dt_nd, 'om')
        else:
            plt.plot(omega_fit / omega_nd, delta_t_fit2 / Dt_nd, '-k', label='Fit ')
            plt.plot(omega_sorted_est / omega_nd, teff_sorted_omega_index_est / Dt_nd, 'om')

        # plt.errorbar(omega_array_devD0_2/omega_nd, delta_t_devD0_2/Dt_nd, yerr=model_error_devD0_2/Dt_nd, fmt=marker_style[3], mec=marker_color[3], uplims=True, lolims=True, label='Dev D CR0_2')
        #a = str(int(np.log10(np.max(omega_sorted_est))))
        #b = str(int(np.log10(np.max(teff_sorted_omega_index_est))))

        xlabel_fig = r'$\omega$ ($10^7\,\mathrm{s}^{-1}$)'
        ylabel_fig = r'Time Offset ($10^{-5}$ s)'
        plt.xlabel(xlabel_fig, labelpad=10)
        plt.ylabel(ylabel_fig, labelpad=10)

        if plot_type == 'loglog':
            plt.text(0.34, 0.06, r'Curve Fit: $\Delta t = \frac{a}{\omega + b \omega^2}$',
                     {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center'})
            plt.text(0.34, 0.05, r'$a = 6.2774 \pm 0.01502$',
                     {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})
            plt.text(0.34, 0.04, r'$b = 1.5086 \times 10^{-9} \pm 0.6104 \times 10^{-9}$ (s)',
                     {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})

            xticks_plot = [0.1, 0.2, 0.5, 1.0, 1.5]
            plt.xticks(xticks_plot)
            plt.gca().set_xticklabels(['0.1', '0.2', '0.5', '1.0', '1.5'])

            yticks_plot = [0.02,0.05,0.1,0.2,0.5]
            plt.yticks(yticks_plot)
            plt.gca().set_yticklabels(['0.02', '0.05', '0.1', '0.2', '0.5'])
        else:
            plt.text(1.0, 0.25, r'Curve Fit: $\Delta t = \frac{a}{\omega + b \omega^2}$',
                     {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center'})
            plt.text(1.0, 0.22, r'$a = 6.2774 \pm 0.01502$',
                     {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})
            plt.text(1.0, 0.19, r'$b = 1.5086 \times 10^{-9} \pm 0.6104 \times 10^{-9}$ (s)',
                     {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})

        plt.legend(loc='upper right')
        plt.savefig(filename_plot, bbox_inches='tight')

    return popt2, np.sqrt(np.diag(pcov2))


def plot_control_noise_model(Dt_array_est, omega_array_est,
                             omega_fit, delta_t_fit2,
                             omega_array_devA_est, delta_t_devA_est, model_error_devA,
                             omega_array_devC_est, delta_t_devC_est, model_error_devC,
                             omega_array_devD0_1_est, delta_t_devD0_1_est, model_error_devD0_1,
                             omega_array_devD0_2_est, delta_t_devD0_2_est, model_error_devD0_2,
                             plot_type='loglog'):

    marker_color = ['b', 'g', 'r', 'c', 'm', 'y']
    marker_style = ['xb', 'xg', 'xr', 'xc', 'xm', 'xy']

    Dt_nd = 10 ** (int(np.log10(np.max(Dt_array_est))))
    omega_nd = 10 ** (int(np.log10(np.max(omega_array_est))))

    plt.figure(6, figsize=(8, 8))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    # Plot Data
    if plot_type == 'loglog':
        plt.loglog(omega_fit / omega_nd, delta_t_fit2 / Dt_nd, '-k', label='Fit ')
    else:
        plt.plot(omega_fit / omega_nd, delta_t_fit2 / Dt_nd, '-k', label='Fit ')

    plt.errorbar(omega_array_devA_est / omega_nd, delta_t_devA_est / Dt_nd, yerr=model_error_devA / Dt_nd,
                 fmt=marker_style[0], mec=marker_color[0], uplims=True, lolims=True, label='Dev A')
    plt.errorbar(omega_array_devC_est / omega_nd, delta_t_devC_est / Dt_nd, yerr=model_error_devC / Dt_nd,
                 fmt=marker_style[1], mec=marker_color[1], uplims=True, lolims=True, label='Dev B')
    plt.errorbar(omega_array_devD0_1_est / omega_nd, delta_t_devD0_1_est / Dt_nd, yerr=model_error_devD0_1 / Dt_nd,
                 fmt=marker_style[2], mec=marker_color[2], uplims=True, lolims=True, label=r'Dev C $\mathrm{CR}0_1$')
    plt.errorbar(omega_array_devD0_2_est / omega_nd, delta_t_devD0_2_est / Dt_nd, yerr=model_error_devD0_2 / Dt_nd,
                 fmt=marker_style[3], mec=marker_color[3], uplims=True, lolims=True, label=r'Dev C $\mathrm{CR}0_2$')
    # xlabel_fig = r'$\omega$ [1e' + str(int(np.log10(np.max(omega_array)))) + ']'
    # ylabel_fig = 'Time Offset [1e' + str(int(np.log10(np.max(Dt_array)))) + ']'

    xlabel_fig = r'$\omega$ ($10^7\,\mathrm{s}^{-1}$)'
    ylabel_fig = r'Time Offset ($10^{-5}$ s)'

    plt.xlabel(xlabel_fig, labelpad=10)
    plt.ylabel(ylabel_fig, labelpad=10)

    if plot_type == 'loglog':
        plt.text(0.5, 0.006, r'Curve Fit: $\Delta t = \frac{a}{\omega + b \omega^2}$',
                 {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center'})
        plt.text(0.5, 0.004, r'$a = 3.1458 \pm 0.02082$',
                 {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})
        plt.text(0.5, 0.0025, r'$b = 1.423 \times 10^{-8} \pm 1.498 \times 10^{-9}$ (s)',
                 {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})

        xticks_plot = [0.1, 0.2, 0.5, 1, 2, 5, 10]
        plt.xticks(xticks_plot)
        plt.gca().set_xticklabels(['0.1', '0.2', '0.5', '1', '2', '5', '10'])

        yticks_plot = [0.001, 0.01, 0.1]
        plt.yticks(yticks_plot)
        plt.gca().set_yticklabels(['0.001', '0.01', '0.1'])
    else:
        plt.text(5.0, 0.11, r'Curve Fit: $\Delta t = \frac{a}{\omega + b \omega^2}$',
                 {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center'})
        plt.text(5.0, 0.09, r'$a = 3.1458 \pm 0.02082$',
                 {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})
        plt.text(5.0, 0.07, r'$b = 1.423 \times 10^{-8} \pm 1.498 \times 10^{-9}$ (s)',
                 {'color': 'black', 'fontsize': 20, 'ha': 'center', 'va': 'center'})

    plt.legend(loc='upper right')
    plt.savefig('control_noise_model_old_ibm_devices_loglog.eps',bbox_inches='tight')


# Functions to calculate the initial estimates of the parameters from Rabi Oscillations for old Devices (A,B,C):
# Function to calculate the Fourier transforms of the p(0)-p(1) estimates
def calc_freq(rabi_data, freq_convert):
    n_configs = rabi_data.shape[0]
    rabi_data_means = np.ndarray(shape=n_configs, dtype=float)
    rabi_data_0mean = np.ndarray(shape=rabi_data.shape, dtype=float)

    # Obtain the zero meaned series of the Rabi Oscillations' data
    for ind_config in range(n_configs):
        rabi_data_means[ind_config] = np.mean(rabi_data[ind_config, :])
        rabi_data_0mean[ind_config, :] = list(map(lambda x: x - rabi_data_means[ind_config], rabi_data[ind_config]))

    # Calculate Fourier transforms of the Rabi Oscillations
    chi_size = (2, rabi_data.shape[1])
    rabi_fourier = np.ndarray(shape=rabi_data_0mean.shape, dtype=complex)
    rabi_power = np.ndarray(shape=rabi_data_0mean.shape, dtype=float)
    rabi_chisq = np.ndarray(shape=chi_size, dtype=float)
    rabi_freq = np.ndarray(shape=(2,), dtype=float)

    for ind_config in range(n_configs):
        rabi_fourier[ind_config, :] = np.fft.fft(rabi_data_0mean[ind_config, :])
        rabi_power[ind_config, :] = list(map(lambda x: np.real(x * np.conjugate(x)), rabi_fourier[ind_config, :]))

    # Combine power series to form chi-squared specra for frequency identification
    rabi_chisq[0, :] = rabi_power[0, :] + rabi_power[2, :] + rabi_power[4, :]
    rabi_chisq[1, :] = rabi_power[1, :] + rabi_power[3, :] + rabi_power[5, :]

    # Calculate initial frequency estimates for each drive configuration
    rabi_freq[0] = freq_convert * np.argmax(rabi_chisq[0, :int((1 + rabi_chisq.shape[1]) / 2)])
    rabi_freq[1] = freq_convert * np.argmax(rabi_chisq[1, :int((1 + rabi_chisq.shape[1]) / 2)])

    return rabi_power, rabi_chisq, rabi_freq


# Functions relevant for obtaining refined estimates of the parameter values
# Function for generating cos & sin time series of a specified frequency in radians/sec
def cos_sin_features(omega, time_stamps):
    cs_array = np.ndarray(shape=(len(time_stamps), 2), dtype=float)

    cs_array[:, 0] = np.cos(omega * time_stamps)
    cs_array[:, 1] = np.sin(omega * time_stamps)

    return cs_array


# Function to calculate squared fitting error given frequency estimate omega
def sumsq_wrt_omega(omega, series_mod2, rabi_data, time_stamps):
    sumsq = 0
    for ind_config in range(rabi_data.shape[0]):
        if (series_mod2 == ind_config % 2):
            cs_array = cos_sin_features(omega, time_stamps)
            reg = LinearRegression().fit(cs_array, rabi_data[ind_config,:])
            a = reg.coef_[0]
            b = reg.coef_[1]
            c = reg.intercept_
            for k in range(cs_array.shape[0]):
                sumsq += (a*cs_array[k,0] + b*cs_array[k,1] + c - rabi_data[ind_config,k]) ** 2
    return sumsq


# Procedure for generating model predictions
def gen_predictions(pvec_data, rabi_freq, time_stamps, gpt):
    rabi_predict = np.ndarray(shape=pvec_data.shape, dtype=float)

    for prep in range(2):
        cg = np.cos(gpt[prep, 0])
        sg = np.sin(gpt[prep, 0])

        cp = np.cos(gpt[prep, 1])
        sp = np.sin(gpt[prep, 1])

        s2t = np.sin(2.0 * gpt[prep, 2])
        hs4t = 0.5 * np.sin(4.0 * gpt[prep, 2])

        s2tcp = s2t * cp
        s2tsp = s2t * sp
        hs4tcp = hs4t * cp
        hs4tsp = hs4t * sp

        ax = hs4tcp * cg + s2tsp * sg
        bx = -hs4tcp * sg + s2tsp * cg
        cx = -hs4tcp

        ay = hs4tsp * cg - s2tcp * sg
        by = -hs4tsp * sg - s2tcp * cg
        cy = -hs4tsp

        az = s2t * s2t * cg
        bz = -s2t * s2t * sg
        cz = 1.0 - s2t * s2t

        coswt = np.cos(rabi_freq[prep] * time_stamps)
        sinwt = np.sin(rabi_freq[prep] * time_stamps)

        rabi_predict[0 + prep, :] = ax * coswt + bx * sinwt + cx
        rabi_predict[2 + prep, :] = ay * coswt + by * sinwt + cy
        rabi_predict[4 + prep, :] = az * coswt + bz * sinwt + cz

    return rabi_predict


def sumsq_wrt_gpt(gpt, prep, est, freq, time_stamps):
    cg = math.cos(gpt[0])
    sg = math.sin(gpt[0])

    cp = math.cos(gpt[1])
    sp = math.sin(gpt[1])

    s2t = math.sin(2.0 * gpt[2])
    hs4t = 0.5 * math.sin(4.0 * gpt[2])

    s2tcp = s2t * cp
    s2tsp = s2t * sp
    hs4tcp = hs4t * cp
    hs4tsp = hs4t * sp

    ax = hs4tcp * cg + s2tsp * sg
    bx = -hs4tcp * sg + s2tsp * cg
    cx = -hs4tcp

    ay = hs4tsp * cg - s2tcp * sg
    by = -hs4tsp * sg - s2tcp * cg
    cy = -hs4tsp

    az = s2t * s2t * cg
    bz = -s2t * s2t * sg
    cz = 1.0 - s2t * s2t

    sumsq = 0.0

    coswt = np.cos(freq[prep] * time_stamps)
    sinwt = np.sin(freq[prep] * time_stamps)
    sumsq += np.sum((ax * coswt + bx * sinwt + cx - est[0 + prep, :]) ** 2)
    sumsq += np.sum((ay * coswt + by * sinwt + cy - est[2 + prep, :]) ** 2)
    sumsq += np.sum((az * coswt + bz * sinwt + cz - est[4 + prep, :]) ** 2)

    return sumsq


def grad_sumsq_wrt_gpt(gpt, prep, est, freq, time_stamps):
    cg = math.cos(gpt[0])
    sg = math.sin(gpt[0])

    cp = math.cos(gpt[1])
    sp = math.sin(gpt[1])

    s2t = math.sin(2.0 * gpt[2])
    tc2t = 2.0 * math.cos(2.0 * gpt[2])
    hs4t = 0.5 * math.sin(4.0 * gpt[2])
    ts4t = 4.0 * hs2t
    tc4t = 2.0 * math.cos(4.0 * gpt[2])

    s2tcp = s2t * cp
    s2tsp = s2t * sp
    tc2tcp = tc2t * cp
    tc2tsp = tc2t * sp

    hs4tcp = hs4t * cp
    hs4tsp = hs4t * sp
    tc4tcp = tc4t * cp
    tc4tsp = tc4t * sp

    ax = hs4tcp * cg + s2tsp * sg
    bx = -hs4tcp * sg + s2tsp * cg
    cx = -hs4tcp

    ay = hs4tsp * cg - s2tcp * sg
    by = -hs4tsp * sg - s2tcp * cg
    cy = -hs4tsp

    az = s2t * s2t * cg
    bz = -s2t * s2t * sg
    cz = 1.0 - s2t * s2t

    grad_g = 0.0
    grad_p = 0.0
    grad_t = 0.0

    coswt = np.cos(freq[prep] * time_stamps)
    sinwt = np.sin(freq[prep] * time_stamps)
    err = 2.0 * (ax * coswt + bx * sinwt + cx - est[0 + prep, :])
    grad_g += np.sum(err * (coswt * (-hs4tcp * sg + s2tsp * cg) + sinwt * (-hs4tcp * cg - s2tsp * sg)))
    grad_p += np.sum(err * (coswt * (-hs4tsp * cg + s2tcp * sg) + sinwt * (hs4tsp * sg + s2tcp * cg) + hs4tsp))
    grad_t += np.sum(err * (coswt * (tc4tcp * cg + tc2tsp * sg) + sinwt * (-tc4tcp * sg + tc2tsp * cg) - tc4tcp))

    err = 2.0 * (ay * coswt + by * sinwt + cy - est[2 + prep, :])
    grad_g += np.sum(err * (coswt * (-hs4tsp * sg - s2tcp * cg) + sinwt * (-hs4tsp * cg + s2tcp * sg)))
    grad_p += np.sum(err * (coswt * (hs4tcp * cg + s2tsp * sg) + sinwt * (-hs4tcp * sg + s2tsp * cg) - hs4tcp))
    grad_t += np.sum(err * (coswt * (tc4tsp * cg - tc2tcp * sg) + sinwt * (-tc4tsp * sg - tc2tcp * cg) - tc4tsp))

    err = 2.0 * (az * coswt + bz * sinwt + cz - est[4 + prep, :])
    grad_g += np.sum(err * (coswt * (-s2t * s2t * sg) + sinwt * (-s2t * s2t * cg)))
    # grad_p += 0
    grad_t += np.sum(err * (coswt * (ts4t * cg) + sinwt * (-ts4t * sg) - ts4t))

    return np.array((grad_g, grad_p, grad_t))


def initial_estimate(syn_data, env_cr=None, A_cr=None, FLAG_syn_data=True):
    # TO DO: Remove FLAG and make more general
    # FLAG_syn_data - indicates if data is being generated from synthetic data or sampling action space

    # Extract relevant parameters
    if FLAG_syn_data:
        if 'device' in syn_data.keys():
            if syn_data['device'] == 'ibmq_boeblingen':
                if 'n_time_stamps' in syn_data.keys():
                    time_stamps = syn_data['time_stamps'][0:int(syn_data['n_time_stamps'])]
                else:
                    time_stamps = syn_data['time_stamps'][0:81]
        else:
            time_stamps = syn_data['time_stamps'][0:-1:6]
    else:
        time_stamps = syn_data['time_stamps'] * A_cr.xi_t

    freq_convert = syn_data['freq_convert']

    # Calculate the rabi oscillations from the given data
    if FLAG_syn_data:
        pvec_data = simulate_nature.rabi_data(syn_data, FLAG_readout_noise=False)
    else:
        pvec_data = A_cr.rabi_data(env_cr)

    # Take fourier transforms of the rabi oscillations
    rabi_power, rabi_chisq, rabi_freq = calc_freq(pvec_data, freq_convert)

    # Calculate and store the initial frequency fit
    rabi_prefit = np.ndarray(shape=pvec_data.shape, dtype=float)

    for ind_config in range(6):
        cs_array = cos_sin_features(rabi_freq[(ind_config % 2)], time_stamps)
        reg = LinearRegression().fit(cs_array, pvec_data[ind_config, :])
        rabi_prefit[ind_config] = reg.predict(cs_array)

    # Refine the initial frequency estimates by fitting to the Rabi Oscillations
    ## INTRODUCE TRY AND CATCH WITH 2*freq_convert below
    res = minimize_scalar(sumsq_wrt_omega,
                          bracket=(rabi_freq[0] - freq_convert,
                                   rabi_freq[0],
                                   rabi_freq[0] + freq_convert),
                          args=(0, pvec_data, time_stamps))
    rabi_freq[0] = res.x

    res = minimize_scalar(sumsq_wrt_omega,
                          bracket=(rabi_freq[1] - freq_convert,
                                   rabi_freq[1],
                                   rabi_freq[1] + freq_convert),
                          args=(1, pvec_data, time_stamps))
    rabi_freq[1] = res.x

    # Calculate and store the fitted curves and corresponding Fourier coefficients
    rabi_cos_sin = np.ndarray(shape=pvec_data.shape, dtype=float)
    rabi_residuals = np.ndarray(shape=pvec_data.shape, dtype=float)
    rabi_abc = np.ndarray(shape=(pvec_data.shape[0], 3), dtype=float)

    for ind_config in range(pvec_data.shape[0]):
        cs_array = cos_sin_features(rabi_freq[ind_config % 2], time_stamps)
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

        if (z0.real > 1.0):
            rabi_gpt[prep, 2] = 0.25 * np.pi
        else:
            rabi_gpt[prep, 2] = (0.25 * np.pi) + 0.5 * np.sign(x0.real * y0.imag) * math.acos(z0.real)

    #
    rabi_refined_gpt = np.ndarray(shape=(2, 3), dtype=float)

    res = minimize(sumsq_wrt_gpt, rabi_gpt[0, :],
                   args=(0, pvec_data, rabi_freq, time_stamps),
                   method="trust-constr",
                   jac=False,  # grad_sumsq_wrt_gpt,
                   bounds=((-math.pi, math.pi),
                           (-math.pi, math.pi),
                           (0, 0.5 * math.pi))
                   )

    rabi_refined_gpt[0, :] = res.x

    res = minimize(sumsq_wrt_gpt, rabi_gpt[1, :],
                   args=(1, pvec_data, rabi_freq, time_stamps),
                   method="trust-constr",
                   jac=False,  # grad_sumsq_wrt_gpt,
                   bounds=((-math.pi, math.pi),
                           (-math.pi, math.pi),
                           (0, 0.5 * math.pi))
                   )
    rabi_refined_gpt[1, :] = res.x

    return rabi_freq, rabi_refined_gpt
