import logging

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

__all__ = ["epsp_slope", "find_epsp_peak"]

logger = logging.getLogger(__name__)


def _find_nearest_index(y, y0):
    return np.argmin(np.abs(y - y0))


def _estimate_ts(t):
    """Estimate sampling interval."""
    dt = t[1:] - t[:-1]
    return np.mean(dt)


def find_epsp_peak(t, y, thre=0.2, dist=0.001, w=0.005):
    """
    Find EPSP peak location.

    Args:
        t (ndarray): Timestamps.
        y (ndarray): Recordings.
        thr (float): EPSP maximum search range.
        dist (float): Minimal time differences between peaks.
        w (float): Required time width of peaks.
    """
    # convert to unit samples
    ts = _estimate_ts(t)
    logger.debug("estimated sampling interval {:.4E}s".format(ts))
    dist, w = dist / ts, w / ts

    peaks, props = find_peaks(y, distance=dist, width=w)
    if len(peaks) == 0:
        logger.info("searching in reversed polarity")
        peaks, props = find_peaks(-y, distance=dist, width=w)

    # identify candidate
    if len(peaks) > 1:
        logger.warning("multiple candidates found, use the first one")
        return peaks[0], {k: v[0] for k, v in props.items()}
    elif len(peaks) == 1:
        return peaks[0], props
    else:
        raise ValueError("unable to find an EPSP signature")


def epsp_slope(t, y, ip, pct=0.2, yf=None, return_pos=False):
    """
    Find EPSP slope.

    Args:
        t (ndarray): Timestamps.
        y (ndarray): Recordings.
        ip (ndarray): EPSP peak index.
        pct (float): Intensity single-sided windowing percentage.
        yf (ndarray, optional): Filtered recordings. 
        return_pos (bool, optional): Return slope extraction details.
    """
    if yf is None:
        yf = y

    ypeak = y[ip]
    ymin, ymax = pct * ypeak, (1 - pct) * ypeak
    logger.info("intensity window [{:.4E}, {:.4E}]".format(ymin, ymax))

    imin, imax = (_find_nearest_index(y[:ip], ymin), _find_nearest_index(y[:ip], ymax))

    logger.info("linreg over @[{}, {}]".format(imin, imax))

    t, y = t[imin : imax + 1], y[imin : imax + 1]
    slope, _, r, _, _ = linregress(t, y)
    logger.info("slope={:4f}, r={:.4f}".format(slope, r))

    if return_pos:
        return slope, r, (t[0], t[-1]), (y[0], y[-1])
    else:
        return slope, r
