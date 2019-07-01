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
path = "../data/01_drug/trial_2.h5"

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

    for index, label, c in zip(indice, labels, ["r", "b", "k"]):
        t, rec, rec_filt = preprocess(index, coarse_crop)

        # visualize data
        ax.plot(t, rec, c, label=label, linewidth=1)

        try:
            # using filtered signal to extract slope
            ipk, _ = find_epsp_peak(t, rec_filt)
            ax.scatter(t[ipk], rec[ipk], marker="o", edgecolors="k", facecolor="none")

            # slope
            slope, r, dt, dy = epsp_slope(t, rec, ipk, yf=rec_filt, return_pos=True)
            ax.plot(dt, dy, "k:", linewidth=1)
        except ValueError:
            logger.error("unable to determine slope")

    # labels
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity (mV)")
    ax.legend()

    plt.savefig("{}.png".format(name), dpi=300)
    plt.waitforbuttonpress()


compare_plot([(11, 123), (244, 254)], ["Control", "DNQX treatment"], "1_ampa_no-ampa")




compare_plot(
    [(11, 123), (441, 476)], ["Control", "Mg$^{2+}$ free ACSF"], "2_ampa_nmda"
)


compare_plot(
    [(441, 476), (500, 537)], ["Mg$^{2+}$ free ACSF", "AP5 treatment"], "3_nmda_no-nmda"
)


compare_plot(
    [(11, 123), (441, 476), (538, -1)], ["Control", "Mg$^{2+}$ free ACSF", "TTX treatment"], "4_ampa_nmda_none"
)

