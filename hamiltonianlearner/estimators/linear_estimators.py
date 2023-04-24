"""
Contains all the functions relevant to Hamiltonian parameter estimation using linear regression:
1. Functions here are typically used after carrying out FFT or with prior frequency estimates
2. Contains functions that carry out FFT using a "regression style" approach as well
"""
import numpy as np
import math
import tensorflow as tf
import cmath
from scipy.optimize import minimize
import scipy.fftpack
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar, curve_fit, leastsq

# other optimization imports
from . import fft_estimators


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


def np_model_free_probabilities(rabi_amplitudes, omega, tvec):
    '''
    Return p(0) given rabi amplitudes, omega, t
    '''
    a = rabi_amplitudes[0]
    b = rabi_amplitudes[1]
    c = rabi_amplitudes[2]

    cos_omega_t = np.cos(omega * tvec)
    sin_omega_t = np.sin(omega * tvec)

    fpvec = 0.5*(a*cos_omega_t + b*sin_omega_t + c + 1)

    return fpvec


def tf_model_free_probabilities(rabi_amplitudes, omega, tvec):
    '''
    Return p(0) given rabi amplitudes, omega, t
    '''
    a = rabi_amplitudes[0]
    b = rabi_amplitudes[1]
    c = rabi_amplitudes[2]

    cos_omega_t = tf.cos(omega * tvec)
    sin_omega_t = tf.sin(omega * tvec)

    fpvec = 0.5*(a*cos_omega_t + b*sin_omega_t + c + 1)

    return fpvec
