"""
Contains all the functions relevant to frequency estimation using FFT:
1. Frequency estimation from rabi oscillations using FFT, window functions, high-resolution FFT
2. See "linear_estimators.py" for functions solving FFT using regression
"""
import numpy as np
import scipy.fftpack
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Function to calculate the Fourier transforms of the p(0)-p(1) estimates
def fft_rabi_data(rabi_data, n_configs, FLAG_demean=False):
    # Calculate Fourier transforms of the Rabi Oscillations
    chi_size = (2, rabi_data.shape[1])
    rabi_fourier = np.ndarray(shape=rabi_data.shape, dtype=complex)
    rabi_power = np.ndarray(shape=rabi_data.shape, dtype=float)
    rabi_chisq = np.ndarray(shape=chi_size, dtype=float)

    if FLAG_demean:
        rabi_data_means = np.ndarray(shape=n_configs, dtype=rabi_data.dtype)
        rabi_data_0mean = np.ndarray(shape=rabi_data.shape, dtype=rabi_data.dtype)

        # Obtain the zero meaned series of the Rabi Oscillations' data
        for ind_config in range(n_configs):
            rabi_data_means[ind_config] = np.mean(rabi_data[ind_config, :])
            rabi_data_0mean[ind_config, :] = list(map(lambda x: x - rabi_data_means[ind_config], rabi_data[ind_config]))

        for ind_config in range(n_configs):
            rabi_fourier[ind_config, :] = np.fft.fft(rabi_data_0mean[ind_config, :])
            rabi_power[ind_config, :] = list(map(lambda x: np.real(x * np.conjugate(x)), rabi_fourier[ind_config, :]))
    else:
        for ind_config in range(n_configs):
            rabi_fourier[ind_config, :] = np.fft.fft(rabi_data[ind_config, :])
            rabi_power[ind_config, :] = list(map(lambda x: np.real(x * np.conjugate(x)), rabi_fourier[ind_config, :]))

    # Combine power series to form chi-squared spectra for frequency identification
    rabi_chisq[0, :] = rabi_power[0, :] + rabi_power[2, :] + rabi_power[4, :]
    rabi_chisq[1, :] = rabi_power[1, :] + rabi_power[3, :] + rabi_power[5, :]

    return rabi_fourier, rabi_power, rabi_chisq


def window_function_rabi_data(rabi_data, type_window='hamming'):
    n_configs = rabi_data.shape[0]
    n_t = rabi_data.shape[1]

    if type_window == 'hamming':
        fft_window = np.hamming(n_t)
    elif type_window == 'hanning':
        fft_window = np.hanning(n_t)
    else:
        fft_window = np.hamming(n_t)

    rabi_data_smooth = np.zeros(shape=rabi_data.shape, dtype=complex)
    for ind_config in range(n_configs):
        rabi_data_smooth[ind_config, :] = rabi_data[ind_config, :] * fft_window

    return rabi_data_smooth


# Function to combine the Fourier transforms of the p(0)-p(1) estimates as desired and get freq estimates
def calc_freq(rabi_data, freq_convert, sample_freq=None, tvec=None, do_plot=False,
              FLAG_window=False, type_window='hamming', FLAG_high_resolution=False, prior_freq=None,
              plotting_options={'FLAG_freq_bin': False, 'prior_freq': [0,5e8]}, omega_truth=None):

    n_configs = rabi_data.shape[0]
    n_t = rabi_data.shape[1]

    if FLAG_window:
        print('Using windowing function!')
        rabi_data_smooth = window_function_rabi_data(rabi_data, type_window=type_window)
        _, rabi_power, rabi_chisq = fft_rabi_data(rabi_data_smooth, n_configs)
    else:
        # Get the usual FFT
        _, rabi_power, rabi_chisq = fft_rabi_data(rabi_data, n_configs)

    # Carry out one more FFT if higher resolution desired
    if FLAG_high_resolution:
        print('Using a higher resolution!')

        if tvec is not None:
            rotating_exponential = np.array([np.exp(-1j*2*np.pi*sample_freq/(2*n_t)*_t) for _t in tvec])
        else:
            rotating_exponential = np.array([np.exp(-1j * 2 * np.pi * 1 / (2 * n_t) * k) for k in range(n_t)])

        #rotating_exponential = np.array([np.exp(-1j*2*np.pi*sample_freq/(2*n_t)*k) for k in range(n_t)])

        rabi_data_rot = np.zeros(shape=rabi_data.shape, dtype=complex)
        for ind_config in range(n_configs):
            rabi_data_rot[ind_config,:] = rabi_data[ind_config,:]*rotating_exponential

        if FLAG_window:
            rabi_data_rot_smooth = window_function_rabi_data(rabi_data_rot, type_window=type_window)
            _, rabi_power_rot, _ = fft_rabi_data(rabi_data_rot_smooth, n_configs)
        else:
            _, rabi_power_rot, _ = fft_rabi_data(rabi_data_rot, n_configs)

        #import pdb; pdb.set_trace()

        # Combine the two different rabi_powers, rabi_chisq
        rabi_power_combined = np.zeros(shape=(n_configs,2*n_t), dtype=float)
        rabi_chisq_combined = np.zeros(shape=(2,2*n_t), dtype=float)

        for ind_config in range(n_configs):
            rabi_power_combined[ind_config, 0:-1:2] = rabi_power[ind_config, :]
            rabi_power_combined[ind_config, 1::2] = rabi_power_rot[ind_config, :]

        # Recompute rabi_chisq
        # Combine power series to form chi-squared specra for frequency identification
        rabi_chisq_combined[0, :] = rabi_power_combined[0, :] + rabi_power_combined[2, :] + rabi_power_combined[4, :]
        rabi_chisq_combined[1, :] = rabi_power_combined[1, :] + rabi_power_combined[3, :] + rabi_power_combined[5, :]

        # Calculate initial frequency estimates for each drive configuration -- IGNORING DC VALUE!
        rabi_freq = np.ndarray(shape=(2,), dtype=float)
        rabi_freq[0] = (freq_convert/2) * (np.argmax(rabi_chisq_combined[0, 1:int((rabi_chisq_combined.shape[1] - 1) / 2)]) + 1)
        rabi_freq[1] = (freq_convert/2) * (np.argmax(rabi_chisq_combined[1, 1:int((rabi_chisq_combined.shape[1] - 1) / 2)]) + 1)

        rabi_power = np.copy(rabi_power_combined)
        rabi_chisq = np.copy(rabi_chisq_combined)

        freq_convert_plot = np.copy(freq_convert/2)
    else:
        # Calculate initial frequency estimates for each drive configuration -- IGNORING DC VALUE!
        rabi_freq = np.ndarray(shape=(2,), dtype=float)
        rabi_freq[0] = freq_convert * (np.argmax(rabi_chisq[0, 1:int((1 + rabi_chisq.shape[1]) / 2)]) + 1)
        rabi_freq[1] = freq_convert * (np.argmax(rabi_chisq[1, 1:int((1 + rabi_chisq.shape[1]) / 2)]) + 1)

        freq_convert_plot = np.copy(freq_convert)

    if do_plot:
        # Plot the Power Spectrum
        plt.figure(2, figsize=(10, 7))
        if plotting_options['FLAG_freq_bin']:
            freq_plot = np.arange(len(rabi_power[0,:]))
            xlabel_plot = 'Frequency Bins'
        else:
            freq_plot = freq_convert_plot*np.arange(len(rabi_power[0,:]))
            xlabel_plot = 'Frequency'

        if plotting_options['prior_freq'] is not None:
            freq_temp = freq_convert_plot * np.arange(len(rabi_power[0, :]))
            ind_cut = np.where(freq_temp > plotting_options['prior_freq'][1])[0][0]
            rabi_power_plot = rabi_power[:,0:ind_cut]
            rabi_chisq_plot = rabi_chisq[:,0:ind_cut]
            freq_plot = freq_plot[0:ind_cut]
        else:
            rabi_power_plot = np.copy(rabi_power)
            rabi_chisq_plot = np.copy(rabi_chisq)

        plt.subplot(311)
        plt.title(r'Blue = Control in $|0\rangle$,  Red = Control in $|1\rangle$')
        plt.plot(freq_plot, rabi_power_plot[0, :], 'bo-')
        plt.plot(freq_plot, rabi_power_plot[1, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        plt.xscale('log')
        plt.ylabel(r'$\langle X \rangle$')

        plt.subplot(312)
        plt.plot(freq_plot, rabi_power_plot[2, :], 'bo-')
        plt.plot(freq_plot, rabi_power_plot[3, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        plt.xscale('log')
        plt.ylabel(r'$\langle Y \rangle$')

        plt.subplot(313)
        plt.plot(freq_plot, rabi_power_plot[4, :], 'bo-')
        plt.plot(freq_plot, rabi_power_plot[5, :], 'r^-')

        if omega_truth is not None:
            plt.axvline(x=omega_truth[0], linestyle='--', c='b')
            plt.axvline(x=omega_truth[1], linestyle='--', c='r')
        plt.xscale('log')
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
        plt.xscale('log')

        plt.show()

        print(rabi_freq)
        if omega_truth is not None:
            print([np.abs(rabi_freq[0]-omega_truth[0])/omega_truth[0],
                   np.abs(rabi_freq[1]-omega_truth[1])/omega_truth[1]])

    results = {'rabi_power': rabi_power, 'rabi_chisq': rabi_chisq, 'rabi_freq': rabi_freq}
    return results


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


def power_wrt_omega(omega, series_mod2, rabi_data, time_stamps):
    power = 0
    for ind_config in range(rabi_data.shape[0]):
        if series_mod2 == ind_config % 2:
            cs_array = cos_sin_features(omega, time_stamps)
            reg = LinearRegression().fit(cs_array, rabi_data[ind_config, :])
            a = reg.coef_[0]
            b = reg.coef_[1]
            c = reg.intercept_
            for k in range(cs_array.shape[0]):
                power += (a*cs_array[k,0] + b*cs_array[k,1] + c) ** 2
    return power


# Plotting the squared error-objective for the frequency
def plot_freqfit(rabi_data, time_stamps, sample_freq=None, freq_est=None, freq_truth=None,
                 FLAG_high_resolution=False, FLAG_corrected_power=False):
    if sample_freq is None:
        first_time_stamp = np.amin(time_stamps)
        last_time_stamp = np.amax(time_stamps)

        delta_time = (last_time_stamp - first_time_stamp) / (len(time_stamps) - 1)
        if FLAG_high_resolution:
            sample_freq = 1.0 / (2*delta_time)
        else:
            sample_freq = 1.0 / delta_time

    if FLAG_high_resolution:
        num_freq_bins = 2000
    else:
        num_freq_bins = 1000

    rabi_freqfit = np.ndarray(shape=(2, num_freq_bins), dtype=float)
    rabi_freqval = np.ndarray(shape=(2, num_freq_bins), dtype=float)

    # Calculate and plot the squared error at discrete frequencies across the spectrum
    for k in range(num_freq_bins):
        # omega = (math.pi * k * samp_freq[0]) / num_freq_bins
        omega = (np.pi * sample_freq) * 10 ** (-(3.0 * (num_freq_bins - k)) / num_freq_bins)
        rabi_freqval[0, k] = omega / (2 * np.pi)

        if FLAG_corrected_power:
            rabi_freqfit[0, k] = power_wrt_omega(omega, 0, rabi_data, time_stamps)
        else:
            rabi_freqfit[0, k] = sumsq_wrt_omega(omega, 0, rabi_data, time_stamps)

        # omega = (math.pi * k * samp_freq[1]) / num_freq_bins
        omega = (np.pi * sample_freq) * 10 ** (-(3.0 * (num_freq_bins - k)) / num_freq_bins)
        rabi_freqval[1, k] = omega / (2 * np.pi)

        if FLAG_corrected_power:
            rabi_freqfit[1, k] = power_wrt_omega(omega, 1, rabi_data, time_stamps)
        else:
            rabi_freqfit[1, k] = sumsq_wrt_omega(omega, 1, rabi_data, time_stamps)

    plt.rcParams['figure.figsize'] = [10.25, 7]
    plt.title(r'Blue = Control in $|0\rangle$,  Red = Control in $|1\rangle$')
    plt.xscale('log')
    plt.plot(rabi_freqval[0, :], rabi_freqfit[0, :], 'b-',
             rabi_freqval[1, :], rabi_freqfit[1, :], 'r-')

    if freq_est is not None:
        plt.axvline(x=freq_est[0], color='b', ls='--', label='Estimate')
        plt.axvline(x=freq_est[1], color='r', ls='--', label='Estimate')

    if freq_truth is not None:
        plt.axvline(x=freq_truth[0], color='b', ls='-', label='Truth')
        plt.axvline(x=freq_truth[1], color='r', ls='-', label='Truth')

    plt.legend(loc='best')
    plt.ylabel('Sum Squared Error')
    plt.xlabel('Frequency')
    plt.show()
