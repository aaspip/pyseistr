#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Yangkang Chen (chenyk2016@gmail.com), 2021-2022   
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# import collections
import logging

class PyseistrError(Exception):
    """
    Base class for all Pyseistr exceptions. Will probably be used for all
    exceptions to not overcomplicate things as the whole package is pretty
    small.
    """
    pass


class PyseistrWarning(UserWarning):
    """
    Base class for all Pyseistr warnings.
    """
    pass

__version__ = "0.1.0"

# Setup the logger.
logger = logging.getLogger("Pyseistr")
logger.setLevel(logging.WARNING)
# Prevent propagating to higher loggers.
logger.propagate = 0
# Console log handler.
ch = logging.StreamHandler()
# Add formatter
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)

from .dip2d import dip2d
from .dip3d import dip3d
from .divne import divne
from .somean2d import somean2d
from .somean3d import somean3d
from .somf2d import somf2d
from .somf3d import somf3d
from .soint3d import soint3d #structure-oriented interpolation for 3D data
from .ricker import ricker
from .bp import bandpass
from .fk import fkdip
from .plot import cseis



## C versions
from .dip2d import dip2dc
from .dip3d import dip3dc
from .somean2d import somean2dc
from .somean3d import somean3dc
from .somf2d import somf2dc
from .somf3d import somf3dc
from .soint3d import soint3dc
from .bp import bandpassc

from dipcfun import *
from sofcfun import *
from sof3dcfun import *
from bpcfun import *





