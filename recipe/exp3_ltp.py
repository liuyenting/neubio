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
    level="debug", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

### filter
fs = 10e3
lo_cutoff = 1e3

### plotter
fig, ax = plt.subplots()


def preprocess(index, crop):
    # load data
    t, stim, rec = load_frame_group(path, index=index, stacked=False)
    # apply filter
    rec_filt = []
    for rec_ in rec:
        rec_ = butter_lpf(rec_, lo_cutoff, fs)
        rec_filt.append(rec_)
    # mean
    rec = np.stack(rec, axis=0).mean(axis=0)
    rec = subtract_baseline(t, rec)
    # filter
    rec_filt = np.stack(rec_filt, axis=0).mean(axis=0)
    rec_filt = subtract_baseline(t, rec_filt)

    # crop
    t_, rec = t_crop(t, rec, crop)
    _, rec_filt = t_crop(t, rec_filt, crop)

    # offset t
    t_ -= t_[0]

    return t_, rec, rec_filt


def compare_plot(indice, labels, name="untitled", **kwargs):
    plt.cla()

    coarse_crop = (0.1, 0.15)
    ax.axhline(0, color="k", linestyle=":", linewidth=0.5)

    for index, label, c in zip(indice, labels, ["r", "b"]):
        t, rec, rec_filt = preprocess(index, coarse_crop)

        # visualize data
        ax.plot(t, rec, c, label=label, linewidth=1)

        try:
            # using filtered signal to extract slope
            ipk, _ = find_epsp_peak(t, rec_filt)
            ax.scatter(
                t[ipk], rec[ipk], marker="o", edgecolors="k", facecolor="none"
            )

            # slope
            slope, r, dt, dy = epsp_slope(t, rec, ipk, yf=rec_filt, return_pos=True)
            ax.plot(dt, dy, "k:", linewidth=1)
        except ValueError:
            logger.error("unable to determine slope")

    # final adjust
    ax.legend()

    plt.savefig("{}.png".format(name), dpi=300)
    plt.waitforbuttonpress()

path = "../data/04_ltp_ltd/trial_4.h5"
compare_plot(
    [(41, 86), (94, 184)], ["Baseline", "Post-HFS"], "ltp_post_hfs"
)
compare_plot(
    [(41, 86), (185, 220)], ["Baseline", "LTP"], "ltp_stable"
)

path = "../data/04_ltp_ltd/trial_5.h5"
compare_plot(
    [(41, 86), (846, 936)], ["Baseline", "Post-LFS"], "ltd_post_lfs"
)
compare_plot(
    [(34, 95), (937, 982)], ["Baseline", "LTD"], "ltd_stable"
)