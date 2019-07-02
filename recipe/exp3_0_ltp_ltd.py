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
    level="warning", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

### filter
fs = 10e3
lo_cutoff = 1e3

### plotter
fig, ax = plt.subplots()


def preprocess(index, crop=(0.1, 0.15)):
    """
    Preprocess data directly from file.

    Args:
        index (ndarray): Frame range to extract, must be contiguous.
        crop (tuple): Timestamp range for coarse croping.
    
    Returns:
        (tuple): tuple containing:
            t (ndarray): Cropped timestamp.
            rec (ndarray): Raw recordings.
            rec_filt (ndarray): Filtered recordings.
    """
    # load data
    t, stim, rec = load_frame_group(path, index=index, stacked=False)

    t_ = None
    rec_tmp, rec_filt = [], []
    for rec_ in rec:
        rec_lpf = butter_lpf(rec_, lo_cutoff, fs)
        rec_lpf = subtract_baseline(t, rec_lpf)
        t_, rec_lpf = t_crop(t, rec_lpf, crop)
        rec_filt.append(rec_lpf)

        rec_ori = subtract_baseline(t, rec_)
        _, rec_ori = t_crop(t, rec_ori, crop)
        rec_tmp.append(rec_ori)
    rec = rec_tmp

    # offset t
    t_ -= t_[0]

    return t_, rec, rec_filt

def extract_amplitude(index, r_min=.7):
    t, rec, rec_filt = preprocess(index)
    
    data = []
    n_discard = 0
    for i, rec_, rec_filt_ in zip(range(index[0], index[1]+1), rec, rec_filt):
        # using filtered signal to extract slope
        ipk, _ = find_epsp_peak(t, rec_filt_)
        # slope
        try:
            slope, r, _, _ = epsp_slope(t, rec_, ipk, yf=rec_filt_, return_pos=True)
        except ValueError:
            n_discard += 1
            logger.warning("error ({})".format(n_discard))

        if abs(r) < r_min:
            n_discard += 1
            logger.warning("discarded new frame ({}), r={:.4f}, ".format(n_discard, r))

        data.append((i, rec_[ipk]))
    x, y = tuple(zip(*data))
    return np.array(x), np.array(y)

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
            ax.scatter(t[ipk], rec[ipk], marker="o", edgecolors="k", facecolor="none")

            # slope
            slope, r, dt, dy = epsp_slope(t, rec, ipk, yf=rec_filt, return_pos=True)
            ax.plot(dt, dy, "k:", linewidth=1)
        except ValueError:
            logger.error("unable to determine slope")

    # final adjust
    ax.legend()

    plt.savefig("{}.png".format(name), dpi=300)
    plt.waitforbuttonpress()

"""
path = "../data/03_ltp_ltd/trial_4.h5"
compare_plot([(41, 86), (94, 184)], ["Baseline", "Post-HFS"], "ltp_post_hfs")
compare_plot([(41, 86), (185, 220)], ["Baseline", "LTP"], "ltp_stable")

path = "../data/03_ltp_ltd/trial_5.h5"
compare_plot([(41, 86), (846, 936)], ["Baseline", "Post-LFS"], "ltd_post_lfs")
compare_plot([(34, 95), (937, 982)], ["Baseline", "LTD"], "ltd_stable")
"""

#####

"""
plt.cla()

path = "../data/03_ltp_ltd/trial_4.h5"
x, y = extract_amplitude((41, 220))

# calculate average line
i = np.argmax(x > 86)
print("BEFORE")
print(".. i={}".format(i))
print(".. n={}".format(len(y[:i])))
amp0 = np.mean(y[:i])
std0 = np.std(y[:i])
print(".. {:.4f} +/- {:4f}".format(amp0, std0))
print()

i = np.argmax(x > 185)
print("AFTER")
print(".. i={}".format(i))
print(".. n={}".format(len(y[i+1:])))
amp1 = np.mean(y[i+1:])
std1 = np.std(y[i+1:])
print(".. {:.4f} +/- {:4f}".format(amp1, std1))
print()

ax.axhline(amp0, color="k", linestyle=":", linewidth=0.5)
ax.axhline(amp1, color="k", linestyle=":", linewidth=0.5)

# adjust timestamp to actual time
x = (x*20)/60

ax.scatter(x, y, marker=".", edgecolors="k", facecolor="none")
ylim = ax.get_ylim()

plt.fill_betweenx(ylim, [86*20/60], [94*20/60], alpha=0.25, edgecolors="none", facecolor="red")
plt.fill_betweenx(ylim, [94*20/60], [184*20/60], alpha=0.25, edgecolors="none", facecolor="black")

# labels
plt.xlabel('Time (min)')
plt.ylabel('Amplitdue (mV)')

# final 
ax.set_ylim(ylim)

plt.savefig("ltp_amp.png", dpi=300)
plt.waitforbuttonpress()
"""

#####

plt.cla()

path = "../data/03_ltp_ltd/trial_5.h5"
x, y = extract_amplitude((34, 982))


# calculate average line
i = np.argmax(x > 86)
print("BEFORE")
print(".. i={}".format(i))
print(".. n={}".format(len(y[:i])))
amp0 = np.mean(y[:i])
std0 = np.std(y[:i])
print(".. {:.4f} +/- {:4f}".format(amp0, std0))
print()

i = np.argmax(x > 937)
print("AFTER")
print(".. i={}".format(i))
print(".. n={}".format(len(y[i+1:])))
amp1 = np.mean(y[i+1:])
std1 = np.std(y[i+1:])
print(".. {:.4f} +/- {:4f}".format(amp1, std1))
print()

ax.axhline(amp0, color="k", linestyle=":", linewidth=0.5)
ax.axhline(amp1, color="k", linestyle=":", linewidth=0.5)

# adjust timestamp to actual time
x = (x*20)/60

ax.scatter(x, y, marker=".", edgecolors="k", facecolor="none")
ylim = ax.get_ylim()
plt.fill_betweenx(ylim, [96*20/60], [846*20/60], alpha=0.25, edgecolors="none", facecolor="red")
plt.fill_betweenx(ylim, [846*20/60], [937*20/60], alpha=0.25, edgecolors="none", facecolor="black")

# labels
plt.xlabel('Time (min)')
plt.ylabel('Slope (mV/s)')

# final 
ax.set_ylim(ylim)

plt.savefig("ltd_slope.png", dpi=300)
plt.waitforbuttonpress()
