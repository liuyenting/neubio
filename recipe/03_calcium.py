import logging
import os

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

coloredlogs.install(
    level="info", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

def load_frames(path, index, group='_frames'):
    start, end = index

    with pd.HDFStore(path) as fd:
        for frame_no in range(start, end+1):
            key = os.path.join(group, str(frame_no))
            df = pd.read_hdf(fd, key)
            yield df
        

def load_frames_numpy(*args, **kwargs):
    xdata, ydata = None, []
    for frame in load_frames(*args, **kwargs):
        if xdata is None:
            xdata = frame['time'].values
        ydata.append(frame['response'].values)
    return xdata, np.stack(ydata, axis=0)

t, y = load_frames_numpy('../data/03_calcium/trial_1.h5', (247, 267))

yavg, yerr = np.mean(y, axis=0), 3*np.std(y, axis=0)

# identify baseline
tmax = .09
i = np.argmax(t>tmax)
yb = np.mean(yavg[:i])
# subtract baseline
yavg -= yb

# crop
tmin, tmax = 0., .5
imin, imax = np.argmax(t>tmin), np.argmax(t>tmax)
t, yavg, yerr = t[imin:imax], yavg[imin:imax], yerr[imin:imax]

fig, ax = plt.subplots()

ax.plot(t, yavg)
ax.fill_between(t, yavg-yerr, yavg+yerr, color='r')
ax.set(xlabel='Time (s)', ylabel='Voltage (mV)',
       title='Paired-pulses')
ax.grid()

plt.show()