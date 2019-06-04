"""
Save HDF5 datasets to other formats.
"""
import logging
import os

import click
import pandas as pd

logger = logging.getLogger(__name__)


@click.command()
@click.argument("path")
@click.argument("key")
@click.option("-v", "--verbose", count=True)
@click.option("-s", "--saveas", default="csv", type=click.Choice(["csv", "plot"]))
@click.option("-o", "--output")
def main(path, key, verbose, saveas, output):
    if verbose == 0:
        verbose = "WARNING"
    elif verbose == 1:
        verbose = "INFO"
    else:
        verbose = "DEBUG"
    coloredlogs.install(
        level="info", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    df = pd.read_hdf(path, key)

    if saveas == "csv":
        pass
    elif saveas == "plot":
        pass
