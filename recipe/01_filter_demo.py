import logging

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np

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
lo_cutoff = 1e3

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

    crop = (0.1, 0.15)

    t_, rec = t_crop(t, rec, crop)
    ax.plot(t_, rec, "r", label="raw", linewidth=1)

    rec_filt = np.stack(rec_filt, axis=0).mean(axis=0)
    rec_filt = subtract_baseline(t, rec_filt)

    t_, rec_filt = t_crop(t, rec_filt, crop)
    ax.plot(t_, rec_filt, "b", label="filtered", linewidth=1)

    ax.axhline(0, color="k", linestyle=":", linewidth=0.5)

    # labels
    ax.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity (mV)")

    # final adjust
    ax.set_xlim(crop)

    plt.savefig("{}.png".format(name), dpi=300)

    plt.waitforbuttonpress()


##### sink #####
sink_index = (39, 64)
main(sink_index, "filter_demo_sink")

##### source #####
source_index = (80, 95)
main(source_index, "filter_demo_source")
