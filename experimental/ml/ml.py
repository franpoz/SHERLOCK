import logging
import os
import shutil
import sys
from multiprocessing import Pool
import pandas as pd
import lightkurve as lk
import foldedleastsquares as tls
import astropy.units as u
import requests
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.mast import Catalogs, Tesscut

from experimental.ml.ml_single_transits_classifier import MLSingleTransitsClassifier

from experimental.ml.ml_model_builder import MLModelBuilder
from experimental.ml.ml_training_set_preparer import MlTrainingSetPreparer
from sherlockpipe.ois.OisManager import OisManager
from lcbuilder.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from lcbuilder.objectinfo.preparer.MissionFfiLightcurveBuilder import MissionFfiLightcurveBuilder
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder
from lcbuilder import eleanor
import numpy as np
import tsfresh
from tsfresh.utilities.dataframe_functions import impute

cpus = 1
first_negative_sector = 1
#ml_training_set_preparer = MlTrainingSetPreparer("training_data/", "/home/martin/")
#ml_training_set_preparer.prepare_positive_training_dataset(cpus)
# #ml_training_set_preparer.prepare_false_positive_training_dataset(cpus)
# ml_training_set_preparer.prepare_negative_training_dataset(first_negative_sector, cpus)
MLSingleTransitsClassifier().load_candidate_single_transits("/mnt/DATA-2/training_data/", "tp")
#MLModelBuilder().get_model()
#MLModelBuilder()get_single_transit_model()
#TODO prepare_negative_training_dataset(negative_dir)
