import pickle5 as pickle
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as scp

# pauli matrices
si = np.array([ [1, 0], [0, 1] ])
sx = np.array([ [0, 1], [1, 0] ])
sy = np.array([ [0, -1j], [1j, 0] ])
sz = np.array([ [1, 0], [0, -1] ])

# hadamard
hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)


def plot_classified_data(classifier, train_data, train_labels, xlim=None, ylim=None,
                         savefilename_fig='binary_classification.pdf',
                         show_plot=True, FLAG_save=False):

    """
    Following the plotting format of PRX
    Ref: https://journals.aps.org/authors/axis-labels-and-scales-on-graphs-h18
    """
    if xlim is None:
        xlim = [np.min(train_data[:, 0]), np.max(train_data[:, 0])]
        xlim = np.around(np.array(xlim) / 5, decimals=0) * 5
        xlim[1] = -65.0

    if ylim is None:
        ylim = [np.min(train_data[:, 1]), np.max(train_data[:, 1])]
        ylim = np.around(np.array(ylim) / 5, decimals=0) * 5
        ylim[0] = -10.0

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 1001),
                         np.linspace(ylim[0], ylim[1], 1001))
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z_grid = Z.reshape(xx.shape)

    x_calib_zero = train_data[train_labels == 0, 0]
    y_calib_zero = train_data[train_labels == 0, 1]

    x_calib_one = train_data[train_labels == 1, 0]
    y_calib_one = train_data[train_labels == 1, 1]

    # To force xlabels and ylabels to use the right defined Font
    # Ref: https://stackoverflow.com/questions/11611374/sans-serif-font-for-axes-tick-labels-with-latex
    from matplotlib import rc
    rc('text.latex', preamble=r'\usepackage{sfmath}')

    plt.figure(figsize=(8, 8))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['text.usetex'] = True

    plt.scatter(x_calib_zero, y_calib_zero, s=8, cmap='viridis', c='blue', alpha=0.5, label=r'$|0 \rangle$')
    plt.scatter(x_calib_one, y_calib_one, s=8, cmap='viridis', c='red', alpha=0.5, label=r'$|1 \rangle$')
    plt.contour(xx, yy, Z_grid, [0.5], colors='k')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('real(c) (arb. units)', labelpad=10)
    plt.ylabel('imag(c) (arb. units)', labelpad=10)
    plt.legend(loc='lower right', shadow=False)

    if FLAG_save is True:
        print('saving file')
        plt.savefig(savefilename_fig, bbox_inches='tight', dpi=600, transparent=True)

    if show_plot:
        plt.show()


def plot_ellipses(gmm, ax, color='red'):
    """
    Source: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
    #sphx-glr-auto-examples-mixture-plot-gmm-covariances-py

    :param gmm:
    :param ax:
    :return:
    """
    for n in range(gmm.n_components):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.3)
        ax.add_artist(ell)
        #ax.set_aspect('equal', 'datalim')


def plot_calibration_gmm(gmm0, gmm1, train_data, train_labels, save_filename='gmm_classification.pdf', FLAG_save=False):
    xlim = [np.min(train_data[:, 0]), np.max(train_data[:, 0])]
    ylim = [np.min(train_data[:, 1]), np.max(train_data[:, 1])]

    xlim[1] = -65.0
    ylim[0] = -10.0

    print(xlim)
    print(ylim)

    x_calib_zero = train_data[train_labels == 0, 0]
    y_calib_zero = train_data[train_labels == 0, 1]

    x_calib_one = train_data[train_labels == 1, 0]
    y_calib_one = train_data[train_labels == 1, 1]

    plt.figure(figsize=(8, 8))
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    #plt.tick_params(axis='both', which='major', pad=15)

    h = plt.subplot(1,1,1)
    plot_ellipses(gmm0, h, color='blue')
    plot_ellipses(gmm1, h, color='red')
    plt.scatter(x_calib_zero, y_calib_zero, s=8, cmap='viridis', c='blue', alpha=0.5, label=r'$|0 \rangle$')
    plt.scatter(x_calib_one, y_calib_one, s=8, cmap='viridis', c='red', alpha=0.5, label=r'$|1 \rangle$')
    #plt.xlim(xlim)
    #plt.ylim(ylim)
    plt.xlim(np.around(np.array(xlim) / 5, decimals=0) * 5)
    plt.ylim(np.around(np.array(ylim) / 5, decimals=0) * 5)
    plt.xlabel('real(c) (arb. units)', labelpad=10)
    plt.ylabel('imag(c) (arb. units)', labelpad=10)
    plt.legend(loc='lower right', shadow=False)

    if FLAG_save is True:
        print('saving file')
        plt.savefig(save_filename, bbox_inches='tight', transparent=True, dpi=600)

    plt.show()


def prob_samples(data):
    """
    This function has been written in haste and probably needs to be updated for future use

    Calculates 2*P(0) - 1 or (1 - 2*P(1)) from classified data

    Assumes that the data is in the form as in make_dataset2
    """
    # KIND OF LIKE A HACK TO ALLOW FOR THE NEW DATA FORMATTING
    if 'device' in data.keys():
        if data['device'] == 'ibmq_boeblingen':
            if 'n_time_stamps' in data.keys():
                time_stamps = data['time_stamps'][0:int(data['n_time_stamps'])]
            else:
                time_stamps = data['time_stamps'][0:81]
    else:
        time_stamps = data['time_stamps'][0:-1:6]

    if data['FLAG_classification'] is True:
        samples = data['samples']
        if 'n_samples_m_u' in data.keys():
            samples = np.reshape(samples, (len(data['time_stamps']), data['n_samples_m_u']))
        else:
            samples = np.reshape(samples, (len(data['time_stamps']), 200))

        prob_samples = 1 - 2 * (np.sum(samples, 1) / samples.shape[1])

    elif data['FLAG_classification'] is False:
        # Calculate the prob_samples by soling a MLE problem
        if 'n_samples_m_u' in data.keys():
            samples_p0_reshaped = np.reshape(data['samples_p0'], (len(data['time_stamps']), data['n_samples_m_u']))
            samples_p1_reshaped = np.reshape(data['samples_p1'], (len(data['time_stamps']), data['n_samples_m_u']))
        else:
            samples_p0_reshaped = np.reshape(data['samples_p0'], (len(data['time_stamps']), 200))
            samples_p1_reshaped = np.reshape(data['samples_p1'], (len(data['time_stamps']), 200))

        # Define the optimization function
        local_MLE_func = lambda p0, samples_p0_row, samples_p1_row : -np.mean(np.log(samples_p0_row*p0 + samples_p1_row*(1-p0)))

        # Define some data structs
        prob_samples = np.zeros(len(data['time_stamps']))

        for ind_row in range(len(data['time_stamps'])):
            samples_p0_row = samples_p0_reshaped[ind_row, :]
            samples_p1_row = samples_p1_reshaped[ind_row, :]

            # Solve the local MLE problem
            results = scp.optimize.minimize(local_MLE_func, 0.5, args=(samples_p0_row, samples_p1_row), bounds=((0,1),) )

            prob_samples[ind_row] = results.x[0]

        prob_samples = 2 * prob_samples - 1

    # KIND OF LIKE A HACK TO ALLOW FOR THE NEW DATA FORMATTING
    pvec = np.zeros((6, len(time_stamps)))
    if 'device' in data.keys():
        if data['device'] == 'ibmq_boeblingen':
            for i in range(6):
                start_row = i*len(time_stamps)
                end_row = (i+1)*len(time_stamps)
                pvec[i, :] = prob_samples[start_row:end_row]
    else:
        for i in range(6):
            pvec[i, :] = prob_samples[i::6]

    return pvec


# Function that defines the values of the pauli coefficients according to Magesan's paper
def scalings_pauli_coeffs(J, Omega, delta1, delta2, Delta):
    a = delta1 + Delta

    Jix = -(J * Omega / a) + (Delta * delta1 * J * Omega ** 3) / ((a ** 3) * (Delta + a) * (2 * a + delta1))

    n1_Jiz = delta1 ** 3 - 2 * delta1 * Delta ** 2 - 2 * Delta ** 3
    d1_Jiz = delta1 * (Delta ** 2) * (a ** 2) * (Delta - delta2)
    n2_Jiz = delta1 ** 2 + Delta ** 2
    d2_Jiz = (Delta ** 2) * (delta2) * (a ** 2)
    n3_Jiz = 6 * delta1 ** 5 + 4 * delta1 ** 4 - 6 * delta1 ** 3 * Delta ** 2 + 7 * delta1 ** 2 * Delta ** 3 + 12 * delta1 * Delta ** 4 + 4 * Delta ** 5
    d3_Jiz = (Delta ** 2) * (a ** 2) * ((delta1 + a) ** 2) * (a + Delta) * (delta1 + 2 * a)
    n4_Jiz = 2
    d4_Jiz = delta1 * a * (a - delta2)
    n5_Jiz = 2
    d5_Jiz = a * (a - delta2) ** 2
    n6_Jiz = 1
    d6_Jiz = Delta * (Delta - delta2) ** 2

    Jiz = ((J ** 2 * Omega ** 2) / 2) * (n1_Jiz / d1_Jiz + n2_Jiz / d2_Jiz + n3_Jiz / d3_Jiz
                                         + n4_Jiz / d4_Jiz + n5_Jiz / d5_Jiz + n6_Jiz / d6_Jiz)

    n2_Jzx = J * (Omega ** 3) * (delta1 ** 2) * (
                3 * delta1 ** 3 + 11 * (delta1 ** 2) * Delta + 15 * delta1 * (Delta ** 2) + 9 * Delta ** 3)
    d2_Jzx = 4 * (Delta ** 3) * (a ** 3) * (a + Delta) * (2 * a + delta1)

    Jzx = -(J * Omega / Delta) * (delta1 / (delta1 + Delta)) + n2_Jzx / d2_Jzx

    n1_Jzz = delta1 ** 3 - 2 * delta1 * Delta ** 2 - 2 * Delta ** 3
    d1_Jzz = delta1 * (Delta ** 2) * (delta2 - Delta)
    f1_Jzz = n1_Jzz / d1_Jzz

    n2_Jzz = 2 * (2 * a + delta1) * (delta1 ** 2 + delta1 * Delta + Delta ** 2)
    d2_Jzz = (Delta ** 2) * (a + delta1) ** 2
    f2_Jzz = n2_Jzz / d2_Jzz

    n3_Jzz = -8 * Delta
    d3_Jzz = 3 * (delta1 ** 2) + 8 * delta1 * Delta + 4 * Delta ** 2
    f3_Jzz = n3_Jzz / d3_Jzz

    f4_Jzz = 2 * delta1 / (Delta * delta2)
    f5_Jzz = -2 * a / ((a - delta2) ** 2)
    f6_Jzz = -2 * a / (delta1 * (a - delta2))
    f7_Jzz = 2 * a * (delta1 + delta2) / (Delta - delta2)

    Jzz = ((J ** 2) / (2 * (delta1 + Delta) ** 2)) * (
                (Omega ** 2) * (f1_Jzz + f2_Jzz + f3_Jzz + f4_Jzz + f5_Jzz + f6_Jzz) + f7_Jzz)

    # Classical cross-talk (Jiy = 0 according to the model but found to be non-zero)
    Jiy = (Jix + Jzx) / 2  # Considered according to Fig. 6 of Magesan's paper

    # Supposed to be zero according to the model but is not according to experiment
    Jzy = Jiz

    scalings = 2 * np.array([Jix, Jiy, Jiz, Jzx, Jzy, Jzz])

    # return scalings in rad/s instead of Hz (as mainly written in the paper)
    return 2*np.pi*scalings


def parameter_sets_expt(n_expt, ham_tom_fits):
    # Define parameter set considering the drive condition interested in
    ham_tom_fit_expt = ham_tom_fits[n_expt]['hamiltonian']['fp']
    Jix = np.pi * ham_tom_fit_expt['IX']
    Jiy = np.pi * ham_tom_fit_expt['IY']
    Jiz = np.pi * ham_tom_fit_expt['IZ']
    Jzx = np.pi * ham_tom_fit_expt['ZX']
    Jzy = np.pi * ham_tom_fit_expt['ZY']
    Jzz = np.pi * ham_tom_fit_expt['ZZ']

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

    J_num = np.array([Jix, Jiy, Jiz, Jzx, Jzy, Jzz])
    param_array = np.array([omega0, delta0, phi0, omega1, delta1, phi1])

    return param_array, J_num


def readout_calibration_classifier(cal_data, cal_labels, samples_expt, nshots_cal, do_plot=True, FLAG_save=False):
    """
    Args:
        cal_data: Implicitly, am assuming that the first row corresponds to |0> and second row corresponds to |1>
        cal_labels:
        samples_expt:
        nshots_cal: number of shots in calibration data of each |0> and |1>
        do_plot:
        FLAG_save:

    Returns:

    Generalize and make this later a part of single-qubit noise models from data
    """
    # Define training data and labels
    train_data_expt = np.reshape(cal_data, -1)
    train_data = np.array([train_data_expt.real, train_data_expt.imag]).T

    train_labels = np.reshape(cal_labels, -1)

    # Define 'testing' data set
    test_data_expt = np.reshape(samples_expt, -1)
    test_data = np.array([test_data_expt.real, test_data_expt.imag]).T

    # apply SVM for classifying the calibration points and accordingly deciding the labels for the signal points

    # Apply Naive Baye's Classifier
    # Initialize our classifier
    gnb = GaussianNB()

    # Train our classifier
    model = gnb.fit(train_data, train_labels)

    # Get misclassification errors
    train_model_labels = gnb.predict(train_data)

    # Error of |0> being misclassified as |1>
    misclassif_error_0_as_1 = 1.0 - accuracy_score(train_labels[:nshots_cal], train_model_labels[:nshots_cal])

    # Error of |1> being misclassified as |0>
    misclassif_error_1_as_0 = 1.0 - accuracy_score(train_labels[nshots_cal:], train_model_labels[nshots_cal:])

    # Apply classifier to samples
    test_labels = gnb.predict(test_data)

    if do_plot:
        plot_classified_data(gnb, train_data, train_labels, FLAG_save=FLAG_save)

        plot_classified_data(gnb, test_data, test_labels)

    return test_labels, misclassif_error_0_as_1, misclassif_error_1_as_0


def readout_calibration_GMM(calib_expt, samples_expt, nshots_cal, n_components=1,
                            do_plot=True, FLAG_save=False, title_fig=None):
    """
    Function to fit GMMs to the single qubit readout calibration data collected

    Args:
        calib_expt: Assumed that the first row corresponds to |0> and second row corresponds to |1>
        test_data:
        n_components:
        do_plot:
        FLAG_save:
        title_fig:

    Returns:

    """
    ## NEED TO UPDATE WITH GMMs (n_components>1) and accordingly the probability calculation
    # Fit GaussianMixture Models to the calibration points
    # GaussianMixture for points read as 0
    train_data0 = np.array([calib_expt[0, :].real, calib_expt[0, :].imag]).T
    labels_train_data0 = np.zeros(nshots_cal)

    gmm0 = GaussianMixture(n_components=1)
    gmm0.fit(train_data0, labels_train_data0)

    # GaussianMixture for points read as 1
    train_data1 = np.array([calib_expt[1,:].real, calib_expt[1,:].imag]).T
    labels_train_data1 = np.ones(nshots_cal)

    gmm1 = GaussianMixture(n_components=1)
    gmm1.fit(train_data1, labels_train_data1)

    # Calculating the probabilities of signal given {0,1}
    mu0 = gmm0.means_[0]  # [1],[2], etc. will be that of the other components
    cov0 = gmm0.covariances_[0]
    mvn0 = multivariate_normal(mu0, cov0)

    mu1 = gmm1.means_[0]
    cov1 = gmm1.covariances_[0]
    mvn1 = multivariate_normal(mu1, cov1)

    # Can this be vectorized?
    test_data_expt = np.reshape(samples_expt, -1)
    test_data = np.array([test_data_expt.real, test_data_expt.imag]).T
    samples_p0 = mvn0.pdf(test_data)  # prob of signal given 0
    samples_p1 = mvn1.pdf(test_data)  # prob of signal given 1

    # Plotting
    if do_plot:
        train_data = np.vstack((train_data0, train_data1))
        train_labels = np.hstack((labels_train_data0, labels_train_data1))
        plot_calibration_gmm(gmm0, gmm1, train_data, train_labels, FLAG_save=FLAG_save)

    return samples_p0, samples_p1


def make_dataset_ibmq_device(data_file_name='ibmq_boel_fixed_qs_data_1.pickle', FLAG_classification=True,
                             do_plot=True, FLAG_save=False):
    """
    Function to create dataset from pickle files dumped from the new device code

    config=0: preparation U0 = II, measurement M1 = IX
    config=1: preparation U1 = XI, measurement M1 = IX
    config=2: preparation U0 = II, measurement M2 = IY
    config=3: preparation U1 = XI, measurement M2 = IY
    config=4: preparation U0 = II, measurement M0 = IZ
    config=5: preparation U1 = XI, measurement M0 = IZ

    :param n_expt:
    :param n_col_samples:
    :param FLAG_classification:
    :param do_plot:
    :return:
    """

    f = open(data_file_name, 'rb')
    data = pickle.load(f)

    # Configurations
    mvec_expt = np.asarray(data['mvec'])
    uvec_expt = np.asarray(data['uvec'])
    tvec_expt = np.asarray(data['time_vec'])

    # IQ Data
    nshots = data['nshots']
    samples_expt = data['offline_data']

    # Configurations for each sample
    mvec_expt = np.tile(mvec_expt, [nshots, 1]).T
    mvec_expt = mvec_expt.reshape(-1)

    uvec_expt = np.tile(uvec_expt, [nshots, 1]).T
    uvec_expt = uvec_expt.reshape(-1)

    # Define some parameters on tvec_expt before changing it
    first_time_stamp = np.amin(tvec_expt)
    last_time_stamp = np.amax(tvec_expt)

    # HACK! -- using the fact that there are 6 different actions of (M,U) for each time t
    n_time_stamps = int(len(tvec_expt) / 6)
    delta_time = (last_time_stamp - first_time_stamp) / (n_time_stamps - 1)
    sample_freq = 1.0 / delta_time
    freq_convert = sample_freq * 2.0 * np.pi / n_time_stamps

    time_stamps_expt = np.copy(tvec_expt)
    tvec_expt = np.tile(tvec_expt, [nshots, 1]).T
    tvec_expt = tvec_expt.reshape(-1)

    # Readout calibration
    cal_data = data['cal_data']
    cal_labels = data['cal_labels']

    # We will classify the data no matter what as we need classified samples for computing a "good" rabi oscillations
    test_labels, r0, r1 = readout_calibration_classifier(cal_data, cal_labels, samples_expt, nshots,
                                                         do_plot=do_plot, FLAG_save=FLAG_save)

    ds = {'mvec': mvec_expt, 'uvec': uvec_expt, 'tvec': tvec_expt, 'time_stamps': time_stamps_expt,
          'samples': test_labels, 'misclassif_error': [r0, r1],
          'n_samples_m_u': nshots, 'FLAG_classification': FLAG_classification, 'first_time_stamp': first_time_stamp,
          'last_time_stamp': last_time_stamp, 'delta_time': delta_time, 'sample_freq': sample_freq,
          'freq_convert': freq_convert, 'device': data['device'], 'n_time_stamps': n_time_stamps}

    # Additionally if requested, we include p(c|obs=0) and p(c|obs=1)
    if FLAG_classification is False:
        samples_p0, samples_p1 = readout_calibration_GMM(cal_data, samples_expt, nshots,
                                                         do_plot=do_plot, FLAG_save=FLAG_save)

        # Include the original IQ data along with p(c|obs)
        ds.update({'samples_iq': samples_expt, 'samples_p0': samples_p0, 'samples_p1': samples_p1})

    return ds


# Used by simulate_nature to create an oracle that is an experimental dataset
def create_queryable_dataset(expt_data):
    """
    Description: This function is meant to be used only for the ibm_data that is processed as above

    The data is of the following form:
    (M, U, t)
    (0, 0, 0): 200 shots
    (0, 1, 0): 200 shots
    (1, 0, 0): 200 shots
    (1, 1, 0): 200 shots
    (2, 0, 0): 200 shots

    :param expt_data:
    :return:
    """
    oracle_expt_data = {}

    # Using knowledge of this fixed query space -- which may need to be changed for other datasets
    for action_i in range(486):
        row_start = action_i*200
        row_end = (action_i+1)*200

        # converting the numpy arrays to lists so easier to pop and delete elements later
        oracle_expt_data.update({action_i: expt_data['samples'][row_start:row_end].tolist()})

    # To keep a count of the number of measurement outcomes available for each action
    oracle_expt_data.update({'n_samples_actions': 200*np.ones(486, dtype=int)})

    return oracle_expt_data


def create_queryable_dataset_ibmq(expt_data, FLAG_classification=False, FLAG_merging=False):
    """
    Description: This function is meant to be used only for the ibm_data that is processed like ibmq_boeblingen

    The data is of the following form:
    (M, U, t)
    (0, 0, 0): 200 shots
    (0, 1, 0): 200 shots
    (1, 0, 0): 200 shots
    (1, 1, 0): 200 shots
    (2, 0, 0): 200 shots

    :param expt_data:
    :return:
    """
    print('Creating query set for ibmq_boeblingen!')
    oracle_expt_data = {}
    #oracle_expt_iq_data = {}
    oracle_expt_p0_data = {}
    oracle_expt_p1_data = {}

    # Classified Samples
    samples = np.reshape(expt_data['samples'], (len(expt_data['time_stamps']), expt_data['n_samples_m_u']))

    if FLAG_classification is False:
        #samples_iq = np.reshape(expt_data['samples_iq'], (len(expt_data['time_stamps']), expt_data['n_samples_m_u']))
        samples_p0 = np.reshape(expt_data['samples_p0'], (len(expt_data['time_stamps']), expt_data['n_samples_m_u']))
        samples_p1 = np.reshape(expt_data['samples_p1'], (len(expt_data['time_stamps']), expt_data['n_samples_m_u']))

    # Permuting the samples in the right way for assignment to the actions as desired
    N_actions = len(expt_data['time_stamps'])
    N_actions_t = int(N_actions/6)

    ind_row_action_permutations = []
    for ind_time in range(N_actions_t):
        ind_row_action_permutations.extend([ind_time + N_actions_t * ind_job for ind_job in range(6)])

    # Using knowledge of this fixed query space -- which may need to be changed for other datasets
    for action_i in range(N_actions):
        ind_row = ind_row_action_permutations[action_i]

        # converting the numpy arrays to lists so easier to pop and delete elements later
        oracle_expt_data.update({action_i: samples[ind_row, :].tolist()})

        if FLAG_classification is False:
            #oracle_expt_iq_data.update({action_i: samples_iq[ind_row, :].tolist()})
            oracle_expt_p0_data.update({action_i: samples_p0[ind_row, :].tolist()})
            oracle_expt_p1_data.update({action_i: samples_p1[ind_row, :].tolist()})

    # To keep a count of the number of measurement outcomes available for each action
    oracle_expt_data.update({'n_samples_actions': expt_data['n_samples_m_u']*np.ones(N_actions, dtype=int)})

    if FLAG_classification is False:
        oracle_expt_data.update({'samples_p0': oracle_expt_p0_data})
        oracle_expt_data.update({'samples_p1': oracle_expt_p1_data})

    # To be able to associate the "queries" with their information
    if FLAG_merging:
        tvec = np.reshape(expt_data['tvec'], (len(expt_data['time_stamps']), expt_data['n_samples_m_u']))[:, 0]
        mvec = np.reshape(expt_data['mvec'], (len(expt_data['time_stamps']), expt_data['n_samples_m_u']))[:, 0]
        uvec = np.reshape(expt_data['uvec'], (len(expt_data['time_stamps']), expt_data['n_samples_m_u']))[:, 0]

        experimental_info = {'time': tvec[ind_row_action_permutations],
                             'uvec': uvec[ind_row_action_permutations],
                             'mvec': mvec[ind_row_action_permutations],
                             'n_queries': len(expt_data['time_stamps'])}

        oracle_expt_data.update({'query_info': experimental_info})
        oracle_expt_data.update({'FLAG_classification': expt_data['FLAG_classification']})
        oracle_expt_data.update({'misclassif_error': expt_data['misclassif_error']})
        oracle_expt_data.update({'device': expt_data['device']})
        oracle_expt_data.update({'freq_convert': expt_data['freq_convert']})

        # HACKY
        oracle_expt_data.update({'FLAG_decoherence': True})

    return oracle_expt_data


def format_device_data(result_job, target, nshots, time_stamps, mvec, uvec, time_vec, moset=None, prepset=None):
    """
    Function to convert the data from result_job that can be used by simulate_nature in it's current form

    Args:
        (For the time being, assuming meas_level=1, can give more options for other things later)
        result_job: of the following form at the moment from looking at how the jobs are setup
        First 2 slots belong to calibration data
        and the next 2 slots are that of
        (M,U,T)
        (0,0,0)
        (0,0,1)
        ...
        (0,0,Nt)
        (0,1,0)
        ...
        (0,1,Nt)
        (1,0,0)
        ...
        (1,0,Nt)
        (1,1,0)
        ...
        (1,1,Nt)
        (2,0,0)
        ...
        (2,0,Nt)
        (2,1,0)
        ...
        (2,1,Nt)

        target: qubit index
        time_stamps: array of all the time_stamps
        moset: dictionary of measurement operators
        prepset: dictionary of preparation operators

        # May want to process later instead of just dumping:
        mvec: array of meas keys from moset
        uvec: array of prep keys from prepset
        tvec: array of normalized times
        time_vec: array of non-normalized times from time_stamps

    Returns: Device data in a format that can be used by process_data with minimal changes to current code
    and simulate_nature (in particular setting up Nature and Action_Space)
    """
    # How much to scale the IQ data by
    scale_factor = 1e-14

    # defaults
    if moset is None:
        moset = {0: [si, scp.linalg.expm(1j * (np.pi / 4) * sy)],
                 1: [si, scp.linalg.expm(-1j * (np.pi / 4) * sx)], 2: [si, si]}

    if prepset is None:
        prepset = {0: [si, si], 1: [sx, si]}

    # calibration data
    cal_data = np.zeros((2,nshots), dtype=complex)
    cal_labels = np.zeros((2,nshots), dtype=int)

    cal_data[0,:] = scale_factor*result_job.get_memory(0)[:,target]
    cal_data[1, :] = scale_factor * result_job.get_memory(1)[:, target]

    cal_labels[1,:] = np.ones(nshots, dtype=int)

    # offline dataset (assuming knowledge of order of (m,u,t) for now)
    offline_data = np.zeros((6*len(time_stamps),nshots), dtype=complex)

    for ind in range(6*len(time_stamps)):
        # ind+2 in memory because the first 2 were calibration data
        offline_data[ind,:] = scale_factor*result_job.get_memory(ind+2)[:,target]

    # Information along with file (there might be a better way to deal with this later)
    ds_info = 'cal - Calibration, offline_data sorted as (M,U,T) think digits, mvec - correspond to rows and not samples, etc.'
    ds = {'cal_data': cal_data, 'cal_labels': cal_labels, 'moset': moset, 'prepset': prepset,
          'offline_data': offline_data, 'nshots': nshots, 'time_stamps': time_stamps,
          'mvec': mvec, 'uvec': uvec, 'time_vec': time_vec,
          'info': ds_info, 'device': 'ibmq_boeblingen'}

    return ds


def format_device_data_chunk(result_job, ind_start, target, nshots, time_stamps, mvec, uvec, time_vec, moset=None, prepset=None):
    """
    Function to convert the data from result_job that can be used by simulate_nature in it's current form

    Args:
        (For the time being, assuming meas_level=1, can give more options for other things later)
        result_job: of the following form at the moment from looking at how the jobs are setup
        First 2 slots belong to calibration data
        and the next 2 slots are that of
        (M,U,T)
        (0,0,0)
        (0,0,1)
        ...
        (0,0,Nt)
        (0,1,0)
        ...
        (0,1,Nt)
        (1,0,0)
        ...
        (1,0,Nt)
        (1,1,0)
        ...
        (1,1,Nt)
        (2,0,0)
        ...
        (2,0,Nt)
        (2,1,0)
        ...
        (2,1,Nt)

        target: qubit index
        time_stamps: array of all the time_stamps
        moset: dictionary of measurement operators
        prepset: dictionary of preparation operators

        # May want to process later instead of just dumping:
        mvec: array of meas keys from moset
        uvec: array of prep keys from prepset
        tvec: array of normalized times
        time_vec: array of non-normalized times from time_stamps

    Returns: Device data in a format that can be used by process_data with minimal changes to current code
    and simulate_nature (in particular setting up Nature and Action_Space)

    """
    # How much to scale the IQ data by
    scale_factor = 1e-14

    # defaults
    if moset is None:
        moset = {0: [si, scp.linalg.expm(1j * (np.pi / 4) * sy)],
                 1: [si, scp.linalg.expm(-1j * (np.pi / 4) * sx)], 2: [si, si]}

    if prepset is None:
        prepset = {0: [si, si], 1: [sx, si]}

    # calibration data
    cal_data = np.zeros((2,nshots), dtype=complex)
    cal_labels = np.zeros((2,nshots), dtype=int)

    cal_data[0,:] = scale_factor*result_job.get_memory(ind_start+0)[:,target]
    cal_data[1, :] = scale_factor * result_job.get_memory(ind_start+1)[:, target]

    cal_labels[1,:] = np.ones(nshots, dtype=int)

    # offline dataset (assuming knowledge of order of (m,u,t) for now)
    offline_data = np.zeros((len(time_stamps),nshots), dtype=complex)

    for ind in range(len(time_stamps)):
        # ind+2 in memory because the first 2 were calibration data
        offline_data[ind,:] = scale_factor*result_job.get_memory(ind_start+ind+2)[:,target]

    # Information along with file (there might be a better way to deal with this later)
    ds_info = 'cal - Calibration, offline_data sorted as (M,U,T) think digits, mvec - correspond to rows and not samples, etc.'
    ds = {'cal_data': cal_data, 'cal_labels': cal_labels, 'moset': moset, 'prepset': prepset,
          'offline_data': offline_data, 'nshots': nshots, 'time_stamps': time_stamps,
          'mvec': mvec, 'uvec': uvec, 'time_vec': time_vec,
          'info': ds_info, 'device': 'ibmq_boeblingen'}

    return ds


def format_device_data_query(result_job, ind_start, ind_query, target, nshots, time_stamps, mvec, uvec, time_vec, moset=None, prepset=None):
    """
    Function to convert the data from result_job that can be used by simulate_nature in it's current form

    Args:
        (For the time being, assuming meas_level=1, can give more options for other things later)
        result_job: of the following form at the moment from looking at how the jobs are setup
        First 2 slots belong to calibration data
        and the next 2 slots are that of
        (M,U,T)
        (0,0,0)
        (0,0,1)
        ...
        (0,0,Nt)
        (0,1,0)
        ...
        (0,1,Nt)
        (1,0,0)
        ...
        (1,0,Nt)
        (1,1,0)
        ...
        (1,1,Nt)
        (2,0,0)
        ...
        (2,0,Nt)
        (2,1,0)
        ...
        (2,1,Nt)

        target: qubit index
        time_stamps: array of all the time_stamps
        moset: dictionary of measurement operators
        prepset: dictionary of preparation operators

        # May want to process later instead of just dumping:
        mvec: array of meas keys from moset
        uvec: array of prep keys from prepset
        tvec: array of normalized times
        time_vec: array of non-normalized times from time_stamps

    Returns: Device data in a format that can be used by process_data with minimal changes to current code
    and simulate_nature (in particular setting up Nature and Action_Space)
    """
    # How much to scale the IQ data by
    scale_factor = 1e-14

    # defaults
    if moset is None:
        moset = {0: [si, scp.linalg.expm(1j * (np.pi / 4) * sy)],
                 1: [si, scp.linalg.expm(-1j * (np.pi / 4) * sx)], 2: [si, si]}

    if prepset is None:
        prepset = {0: [si, si], 1: [sx, si]}

    # calibration data
    cal_data = np.zeros((2,nshots), dtype=complex)
    cal_labels = np.zeros((2,nshots), dtype=int)

    cal_data[0,:] = scale_factor*result_job.get_memory(ind_start+0)[:,target]
    cal_data[1, :] = scale_factor * result_job.get_memory(ind_start+1)[:, target]

    cal_labels[1,:] = np.ones(nshots, dtype=int)

    # offline dataset (assuming knowledge of order of (m,u,t) for now)
    offline_data = np.zeros((1,nshots), dtype=complex)

    offline_data[0,:] = scale_factor*result_job.get_memory(ind_start+ind_query+2)[:,target]

    # Information along with file (there might be a better way to deal with this later)
    ds_info = 'cal - Calibration, offline_data sorted as (M,U,T) think digits, mvec - correspond to rows and not samples, etc.'
    ds = {'cal_data': cal_data, 'cal_labels': cal_labels, 'moset': moset, 'prepset': prepset,
          'offline_data': offline_data, 'nshots': nshots, 'time_stamps': time_stamps,
          'mvec': mvec, 'uvec': uvec, 'time_vec': time_vec,
          'info': ds_info, 'device': 'ibmq_boeblingen'}

    return ds


def shift_queries_dataset(queryable_dataset, shift):
    """
    Quick helper function to shift the queries in the dataset by a certain amount
    """
    shifted_dict = {}

    for ind_query in range(queryable_dataset['query_info']['n_queries']):
        key_temp = ind_query + shift
        shifted_dict.update({key_temp: queryable_dataset[ind_query]})

    shifted_dict.update({'n_samples_actions': queryable_dataset['n_samples_actions']})
    shifted_dict.update({'query_info': queryable_dataset['query_info']})
    return shifted_dict


def merge_ibmq_queryable_datasets(A=0.3, n_jobs=4):
    """
    Quick helper function to merge relevant queryable datasets
    :param A: Drive condition of interest
    :return:

    TODO: Generalize to case of FLAG_classification=False
    """
    # Load pickle file for first job
    ind_job = 0

    # Drive condition or amplitude of interest
    amp = int(100*A)

    print('Using drive of A=%d' % amp)

    pickle_result_filename = "ibmq_boel_adaptive_qs_data_aligned_A_0_%d_meas_1_shots_512_job_%d.pickle" % (amp, ind_job)

    pickle_result_file = 'Data/ibmq_boel/' + pickle_result_filename
    ibm_data = make_dataset_ibmq_device(pickle_result_file, FLAG_classification=True, FLAG_save=False, do_plot=False)
    queryable_ibm_dataset = create_queryable_dataset_ibmq(ibm_data, FLAG_classification=True, FLAG_merging=True)

    for ind_job in range(1, n_jobs):
        # Load pickle file for job
        pickle_result_filename = "ibmq_boel_adaptive_qs_data_aligned_A_0_%d_meas_1_shots_512_job_%d.pickle" % (amp, ind_job)

        pickle_result_file = 'Data/ibmq_boel/' + pickle_result_filename
        X_temp = make_dataset_ibmq_device(pickle_result_file, FLAG_classification=True, FLAG_save=False, do_plot=False)
        queryable_X_temp = create_queryable_dataset_ibmq(X_temp, FLAG_classification=True, FLAG_merging=True)

        # Shift the temp querable dataset
        queryable_X_temp = shift_queries_dataset(queryable_X_temp, queryable_ibm_dataset['query_info']['n_queries'])

        # Merge with present/growing queryable dataset
        n_queries_old = queryable_ibm_dataset['query_info']['n_queries']
        n_queries_new = n_queries_old + queryable_X_temp['query_info']['n_queries']

        for ind_query in range(n_queries_old, n_queries_new):
            queryable_ibm_dataset.update({ind_query: queryable_X_temp[ind_query]})

        # Update n_samples_actions
        key_i = 'n_samples_actions'
        queryable_ibm_dataset.update({key_i: np.concatenate((queryable_ibm_dataset[key_i], queryable_X_temp[key_i]))})

        # Update query_info key
        for key_i in ['mvec', 'uvec', 'time']:
            queryable_ibm_dataset['query_info'].update(
                {key_i: np.concatenate((queryable_ibm_dataset['query_info'][key_i],
                                        queryable_X_temp['query_info'][key_i]))})

        queryable_ibm_dataset['query_info']['n_queries'] += queryable_X_temp['query_info']['n_queries']

    return queryable_ibm_dataset
