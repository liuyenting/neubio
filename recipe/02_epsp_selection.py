import logging

import coloredlogs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
path = "../data/00_epsp/trial_1.h5"

### filter
fs = 10e3
lo_cutoff = 500

### plotter
fig, ax = plt.subplots()


def main(index, name="filter_demo_sink"):
    plt.cla()

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

    # visualize raw data
    crop = (0.1, 0.15)

    t_, rec = t_crop(t, rec, crop)
    ax.plot(t_, rec, "r", label="raw", linewidth=1)

    ax.axhline(0, color="k", linestyle=":", linewidth=0.5)

    # using filtered signal to extract slope
    rec_filt = np.stack(rec_filt, axis=0).mean(axis=0)
    rec_filt = subtract_baseline(t, rec_filt)

    t_, rec_filt = t_crop(t, rec_filt, crop)

    ipk, _ = find_epsp_peak(t_, rec_filt)

    ax.scatter(
        t_[ipk],
        rec_filt[ipk],
        marker="o",
        edgecolors="b",
        facecolor="none",
        label="EPSP peak",
    )

    # slope
    slope, r, dt, dy = epsp_slope(t_, rec, ipk, yf=rec_filt, return_pos=True)

    ax.plot(dt, dy, "b:", label="EPSP slope", linewidth=1)

    # create inset
    axin = inset_axes(ax, width="30%", height="50%", loc=4, borderpad=3)

    t_in, rec_in = t_crop(t_, rec, (0.1025, 0.1100))
    axin.plot(t_in, rec_in, "r", label="raw", linewidth=1)

    axin.scatter(
        t_[ipk],
        rec_filt[ipk],
        marker="o",
        edgecolors="b",
        facecolor="none",
        label="EPSP peak",
    )

    axin.plot(dt, dy, "b:", linewidth=2)

    axin.set_xlim((0.1025, 0.1100))

    # final adjust
    ax.legend()
    ax.set_xlim(crop)

    plt.savefig("{}_slope-{:.3f}_r-{:.3f}.png".format(name, slope, r), dpi=300)

    plt.waitforbuttonpress()


##### sink #####
main((194, 209), "epsp_sel")

