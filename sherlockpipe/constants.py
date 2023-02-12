import os

"""Constants used by SHERLOCK"""

USER_HOME_ELEANOR_CACHE = os.path.join(os.path.expanduser('~'), '.eleanor/')
"""The directory where SHERLOCK will store the eleanor internal data"""
AU_TO_RSUN = 215.032
"""Constant used to convert Astronomical Units to Sun Radius"""
MOMENTUM_DUMP_QUALITY_FLAG = 2 ** 5
"""Quality flag value for momentum dumps"""
