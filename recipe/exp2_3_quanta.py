import logging

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np

from neubio.analyze import find_epsp_peak, epsp_slope
from neubio.filter import butter_lpf, subtract_baseline, t_crop
from neubio.io import load_frame_group

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

coloredlogs.install(
    level="error", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)


### data source
path = "../data/02_calcium/trial_1.h5"

### filter
fs = 10e3
lo_cutoff = 1e3


def preprocess(index):
    # load data
    t, stim, rec = load_frame_group(path, index=index, stacked=False)

    # determine stimuli split point
    ts1 = np.argmax(stim)
    # mask first stimuli
    stim[ts1 : ts1 + 100] = 0
    ts2 = np.argmax(stim)
    ts1, ts2 = t[ts1], t[ts2]
    logger.debug("stimuli timestamp: {}, {}".format(ts1, ts2))

    logger.info("applying LPF and background subtraction")
    # apply filter and subtract baseline
    rec_tmp, rec_filt = [], []
    for rec_ in rec:
        rec_lpf = butter_lpf(rec_, lo_cutoff, fs)
        rec_lpf = subtract_baseline(t, rec_lpf)
        rec_filt.append(rec_lpf)

        rec_ori = subtract_baseline(t, rec_)
        rec_tmp.append(rec_ori)
    rec = rec_tmp

    logger.info("cropping")
    # split stimuli
    t_ = None
    rec1, rec2 = [], []
    rec_filt1, rec_filt2 = [], []
    for rec_, rec_filt_ in zip(rec, rec_filt):
        t_, rec1_ = t_crop(t, rec_, (ts1, ts2))
        rec1.append(rec1_)
        _, rec_filt1_ = t_crop(t, rec_filt_, (ts1, ts2))
        rec_filt1.append(rec_filt1_)

        _, rec2_ = t_crop(t, rec_, (ts2, 2 * ts2 - ts1))
        rec2.append(rec2_)
        _, rec_filt2_ = t_crop(t, rec_filt_, (ts2, 2 * ts2 - ts1))
        rec_filt2.append(rec_filt2_)

    # offset t
    t_ -= t_[0]
    return (
        t_,
        [np.stack(rec1, axis=0), np.stack(rec2, axis=0)],
        [np.stack(rec_filt1, axis=0), np.stack(rec_filt2, axis=0)],
    )


def ppr(index, r_min=.7):
    t, rec, rec_filt = preprocess(index)

    i = 0
    ratio = []
    for rec1_, rec_filt1_, rec2_, rec_filt2_ in zip(
        rec[0], rec_filt[0], rec[1], rec_filt[1]
    ):
        # using filtered signal to extract slope
        ipk, _ = find_epsp_peak(t, rec_filt1_)
        # slope
        _, r, _, _ = epsp_slope(t, rec1_, ipk, yf=rec_filt1_, return_pos=True)
        if abs(r) < r_min:
            i += 1
            logger.warning("discarded new frame ({}), r={:.4f}, ".format(i, r))
            continue
        amp1 = rec1_[ipk]

        # using filtered signal to extract slope
        ipk, _ = find_epsp_peak(t, rec_filt2_)
        # slope
        _, r, _, _ = epsp_slope(t, rec2_, ipk, yf=rec_filt2_, return_pos=True)
        if abs(r) < r_min:
            i += 1
            logger.warning("discarded new frame ({}), r={:.4f}, ".format(i, r))
            continue
        amp2 = rec2_[ipk]
        
        ratio.append(amp2/amp1)

    return np.array(ratio)


mapping = {0.5: (301, 355), 2.5: (247, 300), 5.0: (400, 462)}

for conc, index in mapping.items():
    ratio = ppr(index)

    print("[Ca2+]={}".format(conc))
    print(".. n={}".format(len(ratio)))
    print(".. ratio={:.4f} +/- {:.4f}".format(ratio.mean(), ratio.std()))
    print()
