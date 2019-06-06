import logging
import os

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, freqz

logger = logging.getLogger(__name__)

coloredlogs.install(
    level="info", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)


def load_frames(path, index, group="_frames"):
    start, end = index

    with pd.HDFStore(path) as fd:
        for frame_no in range(start, end + 1):
            key = os.path.join(group, str(frame_no))
            df = pd.read_hdf(fd, key)
            yield df


def load_frames_numpy(*args, **kwargs):
    xdata, ydata = None, []
    for frame in load_frames(*args, **kwargs):
        if xdata is None:
            xdata = frame["time"].values
        ydata.append(frame["response"].values)
    return xdata, np.stack(ydata, axis=0)


# pre
t, y_pre = load_frames_numpy("../data/04_ltp/trial_5.h5", (34, 95))
# post
t, y_post = load_frames_numpy("../data/04_ltp/trial_5.h5", (846, 936))

y_pre, y_post = np.mean(y_pre, axis=0), np.mean(y_post, axis=0)

# LPF
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

order = 2
fs = 10e3
cutoff = 500

y_pre = butter_lowpass_filter(y_pre, cutoff, fs, order)
y_post = butter_lowpass_filter(y_post, cutoff, fs, order)


def subtract_baseline(t, y, tmax=0.09):
    i = np.argmax(t > tmax)
    yb = np.mean(y[:i])
    return y - yb


y_pre = subtract_baseline(t, y_pre)
y_post = subtract_baseline(t, y_post)

# windowing
tmin, tmax = 0.1, 0.2
imin, imax = np.argmax(t > tmin), np.argmax(t > tmax)
t, y_pre, y_post = t[imin:imax], y_pre[imin:imax], y_post[imin:imax]

# cutout volley
i = np.argmax(y_pre)
tmin = t[i+1]
i = np.argmax(t > tmin)
t, y_pre, y_post = t[i:], y_pre[i:], y_post[i:]

# below zero
i = np.argmax(y_pre < 0)
tmin = t[i+1]
i = np.argmax(t > tmin)
t, y_pre, y_post = t[i:], y_pre[i:], y_post[i:]



# slope
dt = np.diff(t, prepend=0)
dy_pre, dy_post = np.diff(y_pre, prepend=0), np.diff(y_post, prepend=0)

i_pre, i_post = np.argmin(dy_pre), np.argmin(dy_post)

print(" pre: {:.5f}".format((dy_pre/dt)[i_pre]))
print("post: {:.5f}".format((dy_post/dt)[i_post]))

fig, ax = plt.subplots(2, 1)

ax[0].plot(t, y_pre, 'b', label="Pre")
ax[0].axvline(x=t[i_pre], color='b', linestyle=':')
ax[0].plot(t, y_post, 'r', label="Post")
ax[0].axvline(x=t[i_post], color='r', linestyle=':')
ax[0].set(xlabel="Time (s)", ylabel="Voltage (mV)", title="LTD")
ax[0].legend()
ax[0].grid()

ax[1].plot(t, dy_pre/dt, 'b', label="Pre")
ax[1].plot(t, dy_post/dt, 'r', label="Post")
ax[1].set(xlabel="Time (s)", ylabel="Slope")
ax[1].legend()
ax[1].grid()

plt.show()
