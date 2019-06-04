import logging

import numpy as np

import neubio

logger = logging.getLogger(__name__)

coloredlogs.install(
    level="info", fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
