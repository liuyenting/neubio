"""
Group datasets into proper experiment subgroups.
"""
from itertools import islice
import logging
import os

import click
import coloredlogs
import h5py
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
def main(verbose):
    if verbose == 0:
        verbose = "WARNING"
    elif verbose == 1:
        verbose = "INFO"
    else:
        verbose = "DEBUG"
    coloredlogs.install(
        level=verbose, fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument("index", type=(int, int))
@click.argument("new_key", type=str, metavar="KEY")
def regroup(path, index, new_key):
    """
    Group datasets in INDEX range into new group KEY. 
    """
    start, end = index
    with h5py.File(path, "r+") as fd:
        if new_key not in fd:
            fd.create_group(new_key)
        for frame_no in range(start, end + 1):
            key = "frame_{}".format(frame_no)
            try:
                fd.move(key, os.path.join(new_key, key))
            except ValueError:
                logger.warning('dataset "{}" does not exists'.format(key))


@main.command()
@click.argument("path")
@click.argument("group")
@click.option("-s", "--start", type=float, default=0)
@click.option("-e", "--end", type=float, default=-1)
def preview(path, group, start, end):
    """
    Preview all the frames in GROUP.
    """
    # fig, ax = plt.subplots()
    # fig.canvas.mpl_connect('key_press_event', press)

    with h5py.File(path, "r") as fd:
        try:
            keys = list(fd[group].keys())
            keys.sort(key=int)
        except KeyError:
            logger.error('unknown group "{}"'.format(group))
            return
        except ValueError:
            logger.info("not a valid frame collection")
            print('"{}" contains these groups:'.format(group))
            for key in keys:
                print("  {}".format(key))
            return

    with pd.HDFStore(path) as fd:
        fig, ax = plt.subplots()
        h, = ax.plot([], [])

        def update_plot(index):
            logger.info(index)

            key = os.path.join(group, keys[index])
            df = pd.read_hdf(fd, key)

            # redraw
            h.set_data(df["time"], df["response"])
            plt.draw()

            return df

        def key_pressed(event):
            index = key_pressed.index

            # update index
            if event.key == "left":
                index -= 1
            elif event.key == "right":
                index += 1
            else:
                return
            if index < 0:
                logger.warning("minimum frame reached")
                index = 0
            elif index > len(keys) - 1:
                logger.warning("maximum frame reached")
                index = len(keys) - 1

            key_pressed.index = index
            update_plot(index)

        # initialize states
        key_pressed.index = 0
        df = update_plot(0)
        xlim, ylim = ((df["time"].min(), df["time"].max()), (-1.5, 1.5))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # attach event callbacks
        fig.canvas.mpl_connect("key_press_event", key_pressed)

        plt.show()
