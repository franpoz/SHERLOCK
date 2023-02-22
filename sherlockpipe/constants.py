"""Constants used by SHERLOCK"""


import os


USER_HOME_ELEANOR_CACHE = os.path.join(os.path.expanduser('~'), '.eleanor/')
"""The directory where SHERLOCK will store the eleanor internal data"""
MOMENTUM_DUMP_QUALITY_FLAG = 2 ** 5
"""Quality flag value for momentum dumps"""
