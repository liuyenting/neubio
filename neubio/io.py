import logging
import os

import pandas as pd

__all__ = ["load_frame_group"]

logger = logging.getLogger(__name__)


def _load_frame_group(path, group="/_frames", index=None):
    with pd.HDFStore(path) as fd:
        # retrieve frame numbers
        _, _, keys = zip(*fd.walk(group))
        keys = sorted(keys[0], key=int)

        try:
            start, end = index
        except TypeError:
            start, end = int(keys[0]), int(keys[-1])
        if end < 0:
            end = int(keys[-1])
        logger.info('loading "{}" ({}->{})'.format(group, start, end))
        ignored = 0
        for frame_no in range(start, end + 1):
            try:
                key = os.path.join(group, str(frame_no))
                yield fd.get(key)
            except KeyError:
                ignored += 1
        if ignored > 0:
            logger.warning("{} frames not found".format(ignored))


def load_frame_group(*args, stacked=True, **kwargs):
    frames = _load_frame_group(*args, **kwargs)
    time, stimuli, response = None, None, []
    for frame in frames:
        if time is None:
            time, stimuli = frame["time"].values, frame["stimuli"].values
        response.append(frame["response"].values)
    if stacked:
        response = np.stack(response, axis=0)
    return time, stimuli, response
