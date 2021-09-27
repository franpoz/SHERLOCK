import logging
import os
import sys

import pandas as pd
import lightkurve as lk
import foldedleastsquares as tls
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.mast import Catalogs, Tesscut
from sherlockpipe.ois.OisManager import OisManager

from lcbuilder.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from lcbuilder.objectinfo.preparer.MissionFfiLightcurveBuilder import MissionFfiLightcurveBuilder
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder
from lcbuilder.eleanor import TargetData
from lcbuilder import eleanor
from lcbuilder.star.TicStarCatalog import TicStarCatalog
import numpy as np
import tsfresh
from tsfresh.utilities.dataframe_functions import impute


def download_neighbours(ID: int, sectors: np.ndarray, search_radius: int = 10):
    """
    Queries TIC for sources near the target and obtains a cutout
    of the pixels enclosing the target.
    Args:
        ID (int): TIC ID of the target.
        sectors (numpy array): Sectors in which the target
                               has been observed.
        search_radius (int): Number of pixels from the target
                             star to search.
    """
    ID = ID
    sectors = sectors
    search_radius = search_radius
    N_pix = 2 * search_radius + 2
    # query TIC for nearby stars
    pixel_size = 20.25 * u.arcsec
    df = Catalogs.query_object(
        str(ID),
        radius=search_radius * pixel_size,
        catalog="TIC"
    )
    new_df = df[
        "ID", "Tmag", "ra", "dec", "mass", "rad", "Teff", "plx"
    ]
    stars = new_df.to_pandas()

    TESS_images = []
    col0s, row0s = [], []
    pix_coords = []
    # for each sector, get FFI cutout and transform RA/Dec into
    # TESS pixel coordinates
    for j, sector in enumerate(sectors):
        Tmag = stars["Tmag"].values
        ra = stars["ra"].values
        dec = stars["dec"].values
        cutout_coord = SkyCoord(ra[0], dec[0], unit="deg")
        cutout_hdu = Tesscut.get_cutouts(cutout_coord, size=N_pix, sector=sector)[0]
        cutout_table = cutout_hdu[1].data
        hdu = cutout_hdu[2].header
        wcs = WCS(hdu)
        TESS_images.append(np.mean(cutout_table["FLUX"], axis=0))
        col0 = cutout_hdu[1].header["1CRV4P"]
        row0 = cutout_hdu[1].header["2CRV4P"]
        col0s.append(col0)
        row0s.append(row0)

        pix_coord = np.zeros([len(ra), 2])
        for i in range(len(ra)):
            RApix = np.asscalar(
                wcs.all_world2pix(ra[i], dec[i], 0)[0]
            )
            Decpix = np.asscalar(
                wcs.all_world2pix(ra[i], dec[i], 0)[1]
            )
            pix_coord[i, 0] = col0 + RApix
            pix_coord[i, 1] = row0 + Decpix
        pix_coords.append(pix_coord)

    # for each star, get the separation and position angle
    # from the targets star
    sep = [0]
    pa = [0]
    c_target = SkyCoord(
        stars["ra"].values[0],
        stars["dec"].values[0],
        unit="deg"
    )
    for i in range(1, len(stars)):
        c_star = SkyCoord(
            stars["ra"].values[i],
            stars["dec"].values[i],
            unit="deg"
        )
        sep.append(
            np.round(
                c_star.separation(c_target).to(u.arcsec).value,
                3
            )
        )
        pa.append(
            np.round(
                c_star.position_angle(c_target).to(u.deg).value,
                3
            )
        )
    stars["sep (arcsec)"] = sep
    stars["PA (E of N)"] = pa

    stars = stars
    TESS_images = TESS_images
    col0s = col0s
    row0s = row0s
    pix_coords = pix_coords
    return stars


dir = "training_data/"
if not os.path.exists(dir):
    os.mkdir(dir)
file_dir = dir + "/ml.log"
if os.path.exists(file_dir):
    os.remove(file_dir)
formatter = logging.Formatter('%(message)s')
logger = logging.getLogger()
while len(logger.handlers) > 0:
    logger.handlers.pop()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
handler = logging.FileHandler(file_dir)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
positive_dir = dir + "/tp/"
negative_dir = dir + "/ntp/"
if not os.path.isdir(dir):
    os.mkdir(dir)
if not os.path.isdir(positive_dir):
    os.mkdir(positive_dir)
if not os.path.isdir(negative_dir):
    os.mkdir(negative_dir)
tp_tic_list = [251848941]
ois = OisManager().load_ois()
ois = ois[(ois["Disposition"] == "CP") | (ois["Disposition"] == "KP")]
mission_lightcurve_builder = MissionLightcurveBuilder()
mission_ffi_lightcurve_builder = MissionFfiLightcurveBuilder()
# TODO fill excluded_ois from given csv file
excluded_ois = {}
# TODO fill additional_ois from given csv file with their ephemeris
additional_ois_df = pd.DataFrame(columns=['Object Id', 'name', 'period', 'period_err', 't0', 'to_err', 'depth',
                                          'depth_err', 'duration', 'duration_err'])
failed_targets_df = pd.DataFrame(columns=["Object Id"])
for tic in ois["Object Id"].unique():
    tic_id = str(tic)
    target_dir = positive_dir + tic_id + "/"
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    lc_short = None
    try:
        logging.info("Trying to get short cadence info for " + tic)
        lcbuild_short = \
            mission_lightcurve_builder.build(MissionObjectInfo(tic_id, 'all'), None)
        lc_short = lcbuild_short.lc
        lc_data = lcbuild_short.lc_data
        lc_data["centroids_x"] = lc_data["centroids_x"] - np.nanmedian(lc_data["centroids_x"])
        lc_data["centroids_y"] = lc_data["centroids_y"] - np.nanmedian(lc_data["centroids_y"])
        lc_data["motion_x"] = lc_data["motion_x"] - np.nanmedian(lc_data["motion_x"])
        lc_data["motion_y"] = lc_data["motion_y"] - np.nanmedian(lc_data["motion_y"])
        lc_data.to_csv(target_dir + "time_series_short.csv")
        tpf_short = lk.search_targetpixelfile(tic_id, cadence="short", author="spoc").download_all()
        short_periodogram = lc_short.to_periodogram(oversample_factor=5)
        periodogram_df = pd.DataFrame(columns=['period', 'power'])
        periodogram_df["period"] = short_periodogram.period.value
        periodogram_df["power"] = short_periodogram.power.value
        periodogram_df.to_csv(target_dir + "periodogram_short.csv")
    except Exception as e:
        logging.warning("No Short Cadence data for target " + tic)
        logging.exception(e)
    logging.info("Trying to get long cadence info for " + tic)
    try:
        lcbuild_long = \
            mission_ffi_lightcurve_builder.build(MissionFfiIdObjectInfo(tic_id, 'all'), None)
        sectors = lcbuild_long.sectors
        lc_long = lcbuild_long.lc
        lc_data = lcbuild_long.lc_data
        lc_data["centroids_x"] = lc_data["centroids_x"] - np.nanmedian(lc_data["centroids_x"])
        lc_data["centroids_y"] = lc_data["centroids_y"] - np.nanmedian(lc_data["centroids_y"])
        lc_data["motion_x"] = lc_data["motion_x"] - np.nanmedian(lc_data["motion_x"])
        lc_data["motion_y"] = lc_data["motion_y"] - np.nanmedian(lc_data["motion_y"])
        lc_data.to_csv(target_dir + "time_series_long.csv")
        lcf_long = lc_long.remove_nans()
        tpf_long = lk.search_targetpixelfile(tic_id, cadence="long", author="spoc").download_all()
        # TODO somehow store tpfs images
        long_periodogram = lc_long.to_periodogram(oversample_factor=5)
        periodogram_df = pd.DataFrame(columns=['period', 'power'])
        periodogram_df["period"] = long_periodogram.period.value
        periodogram_df["power"] = long_periodogram.power.value
        periodogram_df.to_csv(target_dir + "periodogram_long.csv")
        logging.info("Downloading neighbour stars for " + tic)
        stars = download_neighbours(tic, sectors)
        # TODO get neighbours light curves
        logging.info("Classifying candidate points for " + tic)
        target_ois = ois[ois["Object Id"] == tic_id]
        target_ois = target_ois[(target_ois["Disposition"] == "CP") | (target_ois["Disposition"] == "KP")]
        target_ois_df = pd.DataFrame(columns=['id', 'name', 'period', 'period_err', 't0', 'to_err', 'depth', 'depth_err', 'duration', 'duration_err'])
        tags_series_short = np.full(len(lc_short.time), "BL")
        tags_series_long = np.full(len(lc_long.time), "BL")
        for index, row in target_ois.iterrows():
            if row["OI"] not in excluded_ois:
                logging.info("Classifying candidate points with OI %s, period %s, t0 %s and duration %s for " + tic,
                             row["OI"], row["Period (days)"], row["Epoch (BJD)"], row["Duration (hours)"])
                target_ois_df = target_ois_df.append({"id": row["Object Id"], "name": row["OI"], "period": row["Period (days)"],
                                      "period_err": row["Period (days) err"], "t0": row["Epoch (BJD)"] - 2457000.0,
                                      "to_err": row["Epoch (BJD) err"], "depth": row["Depth (ppm)"],
                                      "depth_err": row["Depth (ppm) err"], "duration": row["Duration (hours)"],
                                      "duration_err": row["Duration (hours) err"]}, ignore_index=True)
                if lc_short is not None:
                    mask_short = tls.transit_mask(lc_short.time.value, row["Period (days)"], row["Duration (hours)"] / 24, row["Epoch (BJD)"] - 2457000.0)
                    tags_series_short[mask_short] = "TP"
                mask_long = tls.transit_mask(lc_long.time.value, row["Period (days)"], row["Duration (hours)"] / 24, row["Epoch (BJD)"] - 2457000.0)
                tags_series_long[mask_long] = "TP"
        target_additional_ois_df = additional_ois_df[additional_ois_df["Object Id"] == tic_id]
        for index, row in target_additional_ois_df.iterrows():
            if row["OI"] not in excluded_ois:
                target_ois_df = target_ois_df.append({"id": row["Object Id"], "name": row["OI"], "period": row["Period (days)"],
                                      "period_err": row["Period (days) err"], "t0": row["Epoch (BJD)"] - 2457000.0,
                                      "to_err": row["Epoch (BJD) err"], "depth": row["Depth (ppm)"],
                                      "depth_err": row["Depth (ppm) err"], "duration": row["Duration (hours)"],
                                      "duration_err": row["Duration (hours) err"]}, ignore_index=True)
                if lc_short is not None:
                    mask_short = tls.transit_mask(lc_short.time.value, row["Period (days)"], row["Duration (hours)"] / 24, row["Epoch (BJD)"] - 2457000.0)
                    tags_series_short[mask_short] = "TP"
                mask_long = tls.transit_mask(lc_long.time.value, row["Period (days)"], row["Duration (hours)"] / 24, row["Epoch (BJD)"] - 2457000.0)
                tags_series_long[mask_long] = "TP"
        if lc_short is not None:
            lc_classified_short = pd.DataFrame.from_dict({"time": lc_short.time.value, "flux": lc_short.flux.value, "tag": tags_series_short})
            lc_classified_short.to_csv(target_dir + "lc_classified_short.csv")
        lc_classified_long = pd.DataFrame.from_dict({"time": lc_long.time.value, "flux": lc_long.flux.value, "tag": tags_series_long})
        lc_classified_long.to_csv(target_dir + "lc_classified_long.csv")
        # TODO store folded light curves -with local and global views-(masking previous candidates?)
    except:
        failed_targets_df.append({tic_id})
failed_targets_df.to_csv(dir, index=False)
tsfresh_short_df = pd.DataFrame(columns=['id', 'time', 'flux', 'flux_err', 'background_flux', 'quality', 'centroids_x',
                                   'centroids_y', 'motion_x', 'motion_y'])
tsfresh_long_df = pd.DataFrame(columns=['id', 'time', 'flux', 'flux_err', 'background_flux', 'quality', 'centroids_x',
                                   'centroids_y', 'motion_x', 'motion_y'])
tsfresh_tags_short = []
tsfresh_tags_long = []
for tic_dir in os.listdir(positive_dir):
    short_lc_dir = positive_dir + "/" + tic_dir + "/time_series_short.csv"
    if os.path.exists(short_lc_dir):
        lc_short_df = pd.read_csv(positive_dir + "/" + tic_dir + "/time_series_short.csv")
        lc_short_df['id'] = tic_dir
        tsfresh_short_df.append(lc_short_df)
        tsfresh_tags_short.append([tic_dir, 1])
    lc_long_df = pd.read_csv(positive_dir + "/" + tic_dir + "/time_series_long.csv")
    lc_long_df['id'] = tic_dir
    tsfresh_long_df.append(lc_long_df)
    tsfresh_tags_long.append([tic_dir, 1])
for tic_dir in os.listdir(negative_dir):
    short_lc_dir = negative_dir + "/" + tic_dir + "/time_series_short.csv"
    if os.path.exists(short_lc_dir):
        lc_short_df = pd.read_csv(negative_dir + "/" + tic_dir + "/time_series_short.csv")
        lc_short_df['id'] = tic_dir
        tsfresh_short_df.append(lc_short_df)
        tsfresh_tags_short.append([tic_dir, 1])
    lc_long_df = pd.read_csv(negative_dir + "/" + tic_dir + "/time_series_long.csv")
    lc_long_df['id'] = tic_dir
    tsfresh_long_df.append(lc_long_df)
    tsfresh_tags_long.append([tic_dir, 0])
tsfresh_tags_short = pd.Series(tsfresh_tags_short)
tsfresh_tags_long = pd.Series(tsfresh_tags_long)
# TODO tsfresh needs a dataframe with all the "time series" data (centroids, motion, flux, bck_flux...)
# TODO with an id column specifying the target id and a "y" as a df containing the target ids and the classification
# TODO tag. We need to check how to make this compatible with transit times tagging instead of entire curve
# TODO classification. Maybe https://tsfresh.readthedocs.io/en/latest/text/forecasting.html is helpful.
extracted_features_short = tsfresh.extract_relevant_features(tsfresh_short_df, tsfresh_tags_short, column_id='id',
                                                             column_sort='time')
extracted_features_long = tsfresh.extract_relevant_features(tsfresh_long_df, tsfresh_tags_long, column_id='id',
                                                            column_sort='time')
