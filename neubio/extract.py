import logging
import os
import re

import click
import coloredlogs
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

coloredlogs.install(
    level='DEBUG',
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

def to_csv(root, key, data):
    data = {
        k: v 
        for k, v in data.items()
    }
    df = pd.DataFrame(data, columns=data.keys())

    path = os.path.join(root, '{}.csv'.format(key))
    df.to_csv(path, index=False, header=True)

def to_hdf5(handle, key, data):
    raise NotImplementedError

@click.command()
@click.argument('path')
@click.option('--frame', nargs=2, type=int)
@click.option('--saveas', default='hdf5', type=click.Choice(['hdf5', 'csv']))
def main(path, frame, saveas):
    h_src = h5py.File(path)
    logger.info("{} frames".format(len(h_src)))

    start, end = frame
    dst_root, _ = os.path.splitext(path)
    dst_root = '{}_frame{}-{}'.format(dst_root, start, end)

    if saveas == 'hdf5':
        h_dst = h5py.File(dst_root + '.h5')
    elif saveas == 'csv':
        # create target directory
        try:
            os.mkdir(dst_root)
        except FileExistsError:
            pass
    else:
        raise ValueError("invalid save option")

    # create keys
    keys = ['frame_{}'.format(i) for i in range(start, end+1)] 
    for key in tqdm(keys):
        data = h_src[key]
        if saveas == 'hdf5':
            to_hdf5(h_dst, key, data)
        elif saveas == 'csv':
            to_csv(dst_root, key, data)
