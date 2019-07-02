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


### data source
path = "../data/02_calcium/trial_1.h5"

### filter
fs = 10e3
lo_cutoff = 1e3

### plotter
fig, ax = plt.subplots()


def preprocess(index):
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

    # split data by stimuli
    ts1 = np.argmax(stim)
    # mask
    stim[ts1 : ts1 + 100] = 0
    ts2 = np.argmax(stim)

    ts1, ts2 = t[ts1], t[ts2]
    logger.debug("stimuli timestamp: {}, {}".format(ts1, ts2))

    # split stimuli
    t_, rec1 = t_crop(t, rec, (ts1, ts2))
    _, rec_filt1 = t_crop(t, rec_filt, (ts1, ts2))

    _, rec2 = t_crop(t, rec, (ts2, 2 * ts2 - ts1))
    _, rec_filt2 = t_crop(t, rec_filt, (ts2, 2 * ts2 - ts1))

    # offset t
    t_ -= t_[0]
    return t_, [rec1, rec2], [rec_filt1, rec_filt2]


def dual_pulse_plot(index, name="untitled"):
    plt.cla()

    ax.axhline(0, color="k", linestyle=":", linewidth=0.5)

    t, rec1, rec2 = preprocess(index)

    for rec, rec_filt, c, lbl in zip(
        rec1, rec2, ["r", "b"], ["1st pulse", "2nd pulse"]
    ):
        # visualize data
        ax.plot(t, rec, c, label=lbl, linewidth=1)

        # using filtered signal to extract slope
        ipk, _ = find_epsp_peak(t, rec_filt)
        ax.scatter(
            t[ipk],
            rec[ipk],
            marker="o",
            edgecolors="k",
            facecolor="none"
        )

        # slope
        slope, r, dt, dy = epsp_slope(t, rec, ipk, yf=rec_filt, return_pos=True)
        ax.plot(dt, dy, "k:", linewidth=1)

    # final adjust
    ax.legend()
    # ax.set_xlim(coarse_crop)

    plt.savefig("{}.png".format(name), dpi=300)
    plt.waitforbuttonpress()

dual_pulse_plot((247, 300), "1_2p5_baseline")

#dual_pulse_plot((301, 355), "2_0p5_applied")

#dual_pulse_plot((358, 399), "3_5p0_applied")

#dual_pulse_plot((400, 462), "4_5p0_baseline")
