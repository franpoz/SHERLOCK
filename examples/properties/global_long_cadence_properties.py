import multiprocessing

# This properties file is an example of how to launch Sherlock with a list of mission ids by searching their short
# cadence light curves. We will be analyzing all the given sectors at once.

######################################################################################################################
### GLOBAL OBJECTS RUN SETUP - All sectors analysed at once
######################################################################################################################
GLOBAL_TWO_MIN_IDS = {}
# We will add two mission ids to the pipeline: ONLY TESS objects can be introduced here.
# Note that we can select 'all' the sectors or quarters or specify a subset of them. In this case, the TESS object will
# be processed only using its 13th sector.
GLOBAL_FFI_IDS = {'TIC 299798795': [13], 'TIC 231663901': 'all'}
GLOBAL_FFI_COORDINATES = {}
INPUT_FILES = {}
INPUT_FILES_WITH_IDS = {}

######################################################################################################################
### SECTOR OBJECTS RUN SETUP - All sectors analysed independently
######################################################################################################################
SECTOR_TWO_MIN_IDS = {}
SECTOR_FFI_IDS = {}
SECTOR_FFI_COORDINATES = {}

######################################################################################################################
### DETRENDS SETUP
######################################################################################################################
# If enabled, an initial Savinsky-Golay filter is applied ONLY to SHORT CADENCE light curves (no FFI).
INITIAL_SMOOTH_ENABLED = True
INITIAL_HIGH_RMS_THRESHOLD = 2
INITIAL_HIGH_RMS_BIN_HOURS = 4
DETREND_METHOD = 'biweight'
DETRENDS_NUMBER = 6
DETREND_CORES = multiprocessing.cpu_count() - 1
AUTO_DETREND_ENABLED = True
AUTO_DETREND_METHOD = 'cosine'
AUTO_DETREND_RATIO = 1/4
AUTO_DETREND_PERIOD = None

######################################################################################################################
### TRANSIT ADJUST SETUP
######################################################################################################################
MAX_RUNS = 10
SNR_MIN = 7
SDE_MIN = 5
FAP_MAX = 0.1
CPU_CORES = multiprocessing.cpu_count() - 1
PERIOD_MIN = 0.5
PERIOD_MAX = 33
PERIOD_PROTECT = 10
BIN_MINUTES = 10
MASK_MODE = 'mask'
BEST_SIGNAL_ALGORITHM = 'border-correct'
QUORUM_STRENGTH = 1

######################################################################################################################
### INITIAL MASK
######################################################################################################################
TWO_MIN_MASKS = {}
FFI_IDS_MASKS = {}
FFI_COORDINATES_MASKS = {}
INPUT_FILES_MASKS = {}
INPUT_FILES_WITH_IDS_MASKS = {}

######################################################################################################################
### INITIAL DETREND PERIOD - If any value of this section overlaps any auto-detrend setup, auto-detrend will be
# disabled for the overlapping objects.
######################################################################################################################
TWO_MIN_INITIAL_DETREND_PERIOD = {}
FFI_IDS_INITIAL_DETREND_PERIOD = {}
FFI_COORDINATES_INITIAL_DETREND_PERIOD = {}
INPUT_FILES_INITIAL_DETREND_PERIOD = {}
INPUT_FILES_WITH_IDS_INITIAL_DETREND_PERIOD = {}
