import logging

import numpy as np
from scipy.signal import find_peaks

__all__ = ["epsp_slope"]

logger = logging.getLogger(__name__)


def _find_nearest_index(y, y0):
    return np.argmin(np.abs(y - y0))


def _estimate_ts(t):
    """Estimate sampling interval."""
    dt = t[1:] - t[:-1]
    return np.mean(dt)


def find_epsp(t, y, thre=0.2, dist=0.001, w=0.005, pct=0.2):
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
        return peaks, props
    else:
        raise ValueError("unable to find an EPSP signature")

def epsp_slope(t, y, filter=True, return_info=True):
    """
    Find EPSP slope.

    Args:
        t (ndarray): Timestamps.
        y (ndarray): Recordings.
        thr (float): EPSP maximum search range.
        dist (float): Minimal time differences between peaks.
        w (float): Required time width of peaks.
        pct (float): Intensity single-sided windowing percentage.
    """
    peak, prop = find_epsp(t, y)

    ypeak = y[peak]
    ymin, ymax = pct * ypeak, (1 - pct) * ypeak
    logger.info("intensity window [{:.3f}, {:.3f}]".format(ymin, ymax))

    imin, imax = (
        _find_nearest_index(y[:peak], ymin),
        _find_nearest_index(y[:peak], ymax),
    )

    tmin, tmax = t[imin], t[imax]
    ymin, ymax = y[imin], y[imax]
    slope = (ymax-ymin) / (tmax-tmin)

    if return_info:
        return slope, (tmin, tmax), (ymin, ymax)
    else:
        return slope
