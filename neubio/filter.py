import logging

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

__all__ = ["ac_notch", "butter_hpf", "butter_lpf", "subtract_baseline", "t_crop"]

logger = logging.getLogger(__name__)


def ac_notch(data, fs, f0=60, Q=30.0):
    """
    Notch filter designed for common AC harmonics.

    Args:
        data (ndarray): Input data.
        fs (float): Sampling frequency.
        f0 (float, optional): Frequency to remove. Default to 60 Hz.
        Q (float, optional): Quality factor.
    """
    nyq = 0.5 * fs
    norm_f0 = f0 / nyq
    b, a = iirnotch(norm_f0, Q, fs)
    y = filtfilt(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_hpf(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lpf(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def subtract_baseline(t, y, tmax=0.1):
    """
    Using recordings prior to the stimulus as baseline. Subtract the entire dataseries 
    using that baseline to zero the offset.

    Args:
        t (ndarray): Timestamp array.
        y (ndarray): Recording data.
        tmax (float): Delay till the stimulus occur.
    """
    i = np.argmax(t >= tmax)
    yb = np.median(y[:i])
    return y - yb


def t_crop(t, y, trange):
    """
    Crop recording by timestamp range.

    Args:
        t (ndarray): Timestamp array.
        y (ndarray): Recording data.
        trange: Timestamp range, (start, end).
    """
    try:
        tmin, tmax = trange
        imin, imax = np.argmax(t >= tmin), np.argmax(t >= tmax) + 1
    except ValueError:
        imin, imax = np.argmax(t >= trange), -1
    return t[imin:imax], y[imin:imax]

