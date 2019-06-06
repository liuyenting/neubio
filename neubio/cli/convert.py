"""
Convert Signal3 ASCII files to HDF5.
"""
from io import StringIO
from enum import auto, Enum
import logging
import os
import re

import click
import coloredlogs
import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


__all__ = ["read_signal3"]


def write_frame(fd, frame_no, df):
    """
    Write DataFrame to HDF5.

    Args:
        fd (h5py.File): HDF5 file handle.
        frame_no (int): Frame number.
        df (pandas.DataFrame): Recorded channel data.
    """
    g_name = "_frames/{}".format(frame_no)
    logger.info("writing {}".format(g_name))
    df.to_hdf(fd, g_name)


def scan_for_frames(path, header=r'".*\.cfs","Frame (\d+)"'):
    """
    Scan Signal3 frame structure.

    Args:
        path (str): Signal3 exported ASCII file path.
        header (str): Header regular expression formula.
    
    Yields:
        :rtype: (int, StringIO): Frame number and its extracted raw data string.
    
    Note:
        The raw data string does not contain header.
    """

    class ScannerState(Enum):
        SCANNING = auto()
        FOUND_HEADER = auto()
        COLLECTING = auto()

    state = ScannerState.SCANNING
    frame_no, data = -1, StringIO()
    with open(path, "r") as fd:
        for line in fd:
            if state == ScannerState.SCANNING:
                match = re.match(header, line)
                if match:
                    frame_no = int(match.group(1))
                    logger.debug("frame_{}: start".format(frame_no))
                    state = ScannerState.FOUND_HEADER
            elif state == ScannerState.FOUND_HEADER:
                # transistion state for _1 row header_
                state = ScannerState.COLLECTING
            elif state == ScannerState.COLLECTING:
                if len(line.rstrip()) > 0:
                    data.write(line)
                else:
                    logger.debug("frame_{}: end".format(frame_no))
                    # reset position
                    data.seek(0)
                    yield frame_no, data

                    # empty and reset
                    data.truncate(0)
                    data.seek(0)
                    # restart
                    state = ScannerState.SCANNING


def read_signal3(path, col_def, sep=","):
    """
    Read Signal3 data file.

    Args:
        path (str): Signal3 exported ASCII file path.
        col_def (dict): Desired column name and data format.
        sep (str, optional): Separator used in the file. Default to ','
    
    Yields:
        :rtype: (int, DataFrame): Frame number and its parsed DataFrame.
    """
    logger.debug(path)
    logger.info("reading raw data")

    for frame_no, data in scan_for_frames(path):
        df = pd.read_csv(data, sep=sep, names=col_def.keys(), dtype=col_def)
        yield frame_no, df


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option("-v", "--verbose", count=True)
def main(path, verbose):
    if verbose == 0:
        verbose = "WARNING"
    elif verbose == 1:
        verbose = "INFO"
    else:
        verbose = "DEBUG"
    coloredlogs.install(
        level=verbose, fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    col_def = {"time": np.float32, "response": np.float32, "stimuli": np.float32}
    frames = read_signal3(path, col_def)

    dst_root, _ = os.path.splitext(path)
    path = dst_root + ".h5"
    with pd.HDFStore(path) as fd:
        for frame_no, df in frames:
            write_frame(fd, frame_no, df)
