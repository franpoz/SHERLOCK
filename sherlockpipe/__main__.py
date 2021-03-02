import pickle
import sys

from sherlockpipe.objectinfo.preparer.MissionFfiLightcurveBuilder import MissionFfiLightcurveBuilder
from sherlockpipe.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder

import transitleastsquares
from sherlockpipe.star.starinfo import StarInfo
from sherlockpipe import sherlock
from sherlockpipe.objectinfo.InputObjectInfo import InputObjectInfo
from sherlockpipe.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from sherlockpipe.objectinfo.MissionFfiCoordsObjectInfo import MissionFfiCoordsObjectInfo
from sherlockpipe.objectinfo.MissionInputObjectInfo import MissionInputObjectInfo
from sherlockpipe.objectinfo.MissionObjectInfo import MissionObjectInfo
from argparse import ArgumentParser
from os import path
import yaml
import importlib.util


def load_module(module_path):
    spec = importlib.util.spec_from_file_location("customs", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def get_star_info(properties, id):
    input_star_info = None
    if properties["STAR"] is not None and properties["STAR"][id] is not None:
        star_properties = properties["STAR"][id]
        input_star_info = StarInfo(id, tuple(star_properties["LD_COEFFICIENTS"]) if "LD_COEFFICIENTS" in star_properties else None,
                             star_properties["TEFF"] if "TEFF" in star_properties else None,
                             star_properties["LUM"] if "LUM" in star_properties else None,
                             star_properties["LOGG"] if "LOGG" in star_properties else None,
                             star_properties["RADIUS"] if "RADIUS" in star_properties else None,
                             star_properties["RADIUS_LOWER_ERROR"] if "RADIUS_LOWER_ERROR" in star_properties else None,
                             star_properties["RADIUS_UPPER_ERROR"] if "RADIUS_UPPER_ERROR" in star_properties else None,
                             star_properties["MASS"] if "MASS" in star_properties else None,
                             star_properties["MASS_LOWER_ERROR"] if "MASS_LOWER_ERROR" in star_properties else None,
                             star_properties["MASS_UPPER_ERROR"] if "MASS_UPPER_ERROR" in star_properties else None,
                             star_properties["RA"] if "RA" in star_properties else None,
                             star_properties["DEC"] if "DEC" in star_properties else None)
    return input_star_info

def get_aperture(properties, id):
    input_aperture_file = None
    if properties["APERTURE"] is not None and properties["APERTURE"][id] is not None:
        input_aperture_file = properties["APERTURE"][id]
    return input_aperture_file

def extract_sectors(object_info):
    lc, lc_data, object_star_info, transits_min_count, object_sectors, quarters = \
        MissionFfiLightcurveBuilder().build(object_info, None)
    if object_sectors is not None:
        sections = object_sectors
    else:
        sections = quarters
    return sections


if __name__ == '__main__':
    ap = ArgumentParser(description='Searching for Hints of Exoplanets fRom Lightcurves Of spaCe-based seeKers')
    ap.add_argument('--properties', help="Additional properties to be loaded into Sherlock run ", required=True)
    ap.add_argument('--explore', dest='explore', action='store_true', help="Whether to run using mcmc or ns. Default is ns.")
    args = ap.parse_args()
    resources_dir = path.join(path.dirname(__file__))
    file_dir = resources_dir + "/" + 'properties.yaml' if resources_dir != "" and resources_dir is not None \
        else 'properties.yaml'
    print(resources_dir)
    sherlock_user_properties = yaml.load(open(file_dir), yaml.SafeLoader)
    user_properties = yaml.load(open(args.properties), yaml.SafeLoader)
    sherlock_user_properties.update(user_properties)

    # Build ObjectInfos from properties
    object_infos = []
    mission_object_infos = []
    ffi_object_infos = []
    ffi_coords_object_infos = []
    input_object_infos = []
    input_id_object_infos = []

    ## Adding by-sector analysis objects
    if sherlock_user_properties["SECTOR_TWO_MIN_IDS"]:
        for two_min_id, sectors in sherlock_user_properties["SECTOR_TWO_MIN_IDS"].items():
            star_info = get_star_info(sherlock_user_properties, two_min_id)
            aperture = get_aperture(sherlock_user_properties, two_min_id)
            if sectors == 'all':
                sectors = extract_sectors(MissionObjectInfo(two_min_id, sectors, star_info=star_info, aperture_file=aperture))
                for sector in sectors:
                    mission_object_infos.append(MissionObjectInfo(two_min_id, [sector], star_info=star_info, aperture_file=aperture))
            else:
                for sector in sectors:
                    mission_object_infos.append(MissionObjectInfo(two_min_id, [sector], star_info=star_info, aperture_file=aperture))
    if sherlock_user_properties["SECTOR_FFI_IDS"]:
        for ffi_id, sectors in sherlock_user_properties["SECTOR_FFI_IDS"].items():
            star_info = get_star_info(sherlock_user_properties, None)
            aperture = get_aperture(sherlock_user_properties, ffi_id)
            if sectors == 'all':
                sectors = extract_sectors(MissionFfiIdObjectInfo(ffi_id, sectors, star_info=star_info,
                                                                 aperture_file=aperture))
                for sector in sectors:
                    ffi_object_infos.append(MissionFfiIdObjectInfo(ffi_id, [sector], star_info=star_info,
                                                                   aperture_file=aperture))
            else:
                for sector in sectors:
                    ffi_object_infos.append(MissionFfiIdObjectInfo(ffi_id, [sector], star_info=star_info,
                                                                   aperture_file=aperture))
    if sherlock_user_properties["SECTOR_FFI_COORDINATES"]:
        for coords, sectors in sherlock_user_properties["SECTOR_FFI_COORDINATES"].items():
            star_info = get_star_info(sherlock_user_properties, str(coords[0]) + "_" + str(coords[1]))
            aperture = get_aperture(sherlock_user_properties, coords)
            if sectors == 'all':
                sectors = extract_sectors(MissionFfiCoordsObjectInfo(coords[0], coords[1], sectors, star_info=star_info,
                                                                     aperture_file=aperture))
                for sector in sectors:
                    ffi_object_infos.append(MissionFfiCoordsObjectInfo(coords[0], coords[1], [sector],
                                                                       star_info=star_info, aperture_file=aperture))
            else:
                for sector in sectors:
                    ffi_coords_object_infos.append(MissionFfiCoordsObjectInfo(coords[0], coords[1], [sector], star_info=star_info, aperture_file=aperture))

    ## Adding global analysis objects
    if sherlock_user_properties["GLOBAL_TWO_MIN_IDS"]:
        [mission_object_infos.append(MissionObjectInfo(two_min_id, sectors,
                                                       star_info=get_star_info(sherlock_user_properties, two_min_id),
                                                       aperture_file=get_aperture(sherlock_user_properties, two_min_id)))
         for two_min_id, sectors in sherlock_user_properties["GLOBAL_TWO_MIN_IDS"].items()]
    if sherlock_user_properties["GLOBAL_FFI_IDS"]:
        [ffi_object_infos.append(MissionFfiIdObjectInfo(ffi_id, sectors,
                                                        star_info=get_star_info(sherlock_user_properties, ffi_id),
                                                       aperture_file=get_aperture(sherlock_user_properties, ffi_id)))
         for ffi_id, sectors in sherlock_user_properties["GLOBAL_FFI_IDS"].items()]
    if sherlock_user_properties["GLOBAL_FFI_COORDINATES"]:
        [ffi_coords_object_infos.append(MissionFfiCoordsObjectInfo(coords[0], coords[1], sectors,
                                                       star_info=get_star_info(sherlock_user_properties, str(coords[0]) + "_" + str(coords[1])),
                                                       aperture_file=get_aperture(sherlock_user_properties, coords)))
         for coords, sectors in sherlock_user_properties["GLOBAL_FFI_COORDINATES"].items()]
    if sherlock_user_properties["INPUT_FILES_WITH_IDS"]:
        [input_object_infos.append(
            MissionInputObjectInfo(input_id, sherlock_user_properties["INPUT_FILES_WITH_IDS"][input_id],
                                   star_info=get_star_info(sherlock_user_properties, input_id)))
         for input_id in sherlock_user_properties["INPUT_FILES_WITH_IDS"].keys()]
    if sherlock_user_properties["INPUT_FILES"]:
        [input_id_object_infos.append(InputObjectInfo(file, star_info=get_star_info(sherlock_user_properties, file)))
         for file in sherlock_user_properties["INPUT_FILES"]]

    ## Set mask to object infos
    if sherlock_user_properties["TWO_MIN_MASKS"]:
        for object_info in mission_object_infos:
            if object_info.mission_id() in sherlock_user_properties["TWO_MIN_MASKS"].keys():
                object_info.initial_mask = sherlock_user_properties["TWO_MIN_MASKS"][object_info.mission_id()]
    if sherlock_user_properties["FFI_IDS_MASKS"]:
        for object_info in ffi_object_infos:
            if object_info.mission_id() in sherlock_user_properties["FFI_IDS_MASKS"].keys():
                object_info.initial_mask = sherlock_user_properties["FFI_IDS_MASKS"][object_info.mission_id()]
    if sherlock_user_properties["FFI_COORDINATES_MASKS"]:
        for object_info in ffi_coords_object_infos:
            key = str(object_info.ra) + "_" + str(object_info.dec)
            if key in sherlock_user_properties["FFI_COORDINATES_MASKS"].keys():
                object_info.initial_mask = sherlock_user_properties["FFI_COORDINATES_MASKS"][key]
    if sherlock_user_properties["INPUT_FILES_MASKS"]:
        for object_info in input_object_infos:
            if object_info.input_file in sherlock_user_properties["INPUT_FILES_MASKS"].keys():
                object_info.initial_mask = sherlock_user_properties["INPUT_FILES_MASKS"][object_info.input_file]
    if sherlock_user_properties["INPUT_FILES_WITH_IDS_MASKS"]:
        for object_info in input_id_object_infos:
            if object_info.mission_id() in sherlock_user_properties["INPUT_FILES_WITH_IDS_MASKS"].keys():
                object_info.initial_mask = sherlock_user_properties["INPUT_FILES_WITH_IDS_MASKS"][object_info.mission_id()]

    ## Set transit mask to object infos
    if sherlock_user_properties["TWO_MIN_TRANSIT_MASKS"]:
        for object_info in mission_object_infos:
            if object_info.mission_id() in sherlock_user_properties["TWO_MIN_TRANSIT_MASKS"].keys():
                object_info.initial_transit_mask = sherlock_user_properties["TWO_MIN_TRANSIT_MASKS"][object_info.mission_id()]
    if sherlock_user_properties["FFI_IDS_TRANSIT_MASKS"]:
        for object_info in ffi_object_infos:
            if object_info.mission_id() in sherlock_user_properties["FFI_IDS_TRANSIT_MASKS"].keys():
                object_info.initial_transit_mask = sherlock_user_properties["FFI_IDS_TRANSIT_MASKS"][object_info.mission_id()]
    if sherlock_user_properties["FFI_COORDINATES_TRANSIT_MASKS"]:
        for object_info in ffi_coords_object_infos:
            key = str(object_info.ra) + "_" + str(object_info.dec)
            if key in sherlock_user_properties["FFI_COORDINATES_TRANSIT_MASKS"].keys():
                object_info.initial_transit_mask = sherlock_user_properties["FFI_COORDINATES_TRANSIT_MASKS"][key]
    if sherlock_user_properties["INPUT_FILES_TRANSIT_MASKS"]:
        for object_info in input_object_infos:
            if object_info.input_file in sherlock_user_properties["INPUT_FILES_TRANSIT_MASKS"].keys():
                object_info.initial_transit_mask = sherlock_user_properties["INPUT_FILES_TRANSIT_MASKS"][object_info.input_file]
    if sherlock_user_properties["INPUT_FILES_WITH_IDS_TRANSIT_MASKS"]:
        for object_info in input_id_object_infos:
            if object_info.mission_id() in sherlock_user_properties["INPUT_FILES_WITH_IDS_TRANSIT_MASKS"].keys():
                object_info.initial_transit_mask = sherlock_user_properties["INPUT_FILES_WITH_IDS_TRANSIT_MASKS"][
                    object_info.mission_id()]

    ## Set detrend period to object infos
    if sherlock_user_properties["TWO_MIN_INITIAL_DETREND_PERIOD"]:
        for object_info in mission_object_infos:
            if object_info.mission_id() in sherlock_user_properties["TWO_MIN_INITIAL_DETREND_PERIOD"].keys():
                object_info.initial_detrend_period = sherlock_user_properties["TWO_MIN_INITIAL_DETREND_PERIOD"][object_info.mission_id()]
    if sherlock_user_properties["FFI_IDS_INITIAL_DETREND_PERIOD"]:
        for object_info in ffi_object_infos:
            if object_info.mission_id() in sherlock_user_properties["FFI_IDS_INITIAL_DETREND_PERIOD"].keys():
                object_info.initial_detrend_period = sherlock_user_properties["FFI_IDS_INITIAL_DETREND_PERIOD"][object_info.mission_id()]
    if sherlock_user_properties["FFI_COORDINATES_INITIAL_DETREND_PERIOD"]:
        for object_info in ffi_coords_object_infos:
            key = str(object_info.ra) + "_" + str(object_info.dec)
            if key in sherlock_user_properties["FFI_COORDINATES_INITIAL_DETREND_PERIOD"].keys():
                object_info.initial_detrend_period = sherlock_user_properties["FFI_COORDINATES_INITIAL_DETREND_PERIOD"][key]
    if sherlock_user_properties["INPUT_FILES_INITIAL_DETREND_PERIOD"]:
        for object_info in input_object_infos:
            if object_info.input_file in sherlock_user_properties["INPUT_FILES_INITIAL_DETREND_PERIOD"].keys():
                object_info.initial_detrend_period = sherlock_user_properties["INPUT_FILES_INITIAL_DETREND_PERIOD"][
                    object_info.input_file]
    if sherlock_user_properties["INPUT_FILES_WITH_IDS_INITIAL_DETREND_PERIOD"]:
        for object_info in input_id_object_infos:
            if object_info.mission_id() in sherlock_user_properties["INPUT_FILES_WITH_IDS_INITIAL_DETREND_PERIOD"].keys():
                object_info.initial_detrend_period = \
                sherlock_user_properties["INPUT_FILES_WITH_IDS_INITIAL_DETREND_PERIOD"][object_info.mission_id()]
    user_search_zone = None
    user_prepare = None
    user_selection_algorithm = None
    user_transit_template = None
    from pathlib import Path
    if sherlock_user_properties["CUSTOM_SEARCH_ZONE"] is not None:
        user_search_zone = load_module(sherlock_user_properties["CUSTOM_SEARCH_ZONE"])
        class_name = Path(sherlock_user_properties["CUSTOM_SEARCH_ZONE"].replace(".py", "")).name
        user_search_zone = getattr(user_search_zone, class_name)
        globals()[class_name] = user_search_zone
        pickle.dumps(user_search_zone)
        user_search_zone = user_search_zone()
    if sherlock_user_properties["PREPARE_ALGORITHM"] is not None:
        user_prepare = load_module(sherlock_user_properties["PREPARE_ALGORITHM"])
        class_name = Path(sherlock_user_properties["PREPARE_ALGORITHM"].replace(".py", "")).name
        user_prepare = getattr(user_prepare, class_name)()
        globals()[class_name] = user_prepare
        pickle.dumps(user_prepare)
    if sherlock_user_properties["CUSTOM_SELECTION_ALGORITHM"] is not None:
        user_selection_algorithm = load_module(sherlock_user_properties["CUSTOM_SELECTION_ALGORITHM"])
        class_name = Path(sherlock_user_properties["CUSTOM_SELECTION_ALGORITHM"].replace(".py", "")).name
        user_selection_algorithm = getattr(user_selection_algorithm, class_name)()
        globals()[class_name] = user_selection_algorithm
        pickle.dumps(user_selection_algorithm)
    if sherlock_user_properties["CUSTOM_TRANSIT_TEMPLATE"] is not None:
        user_transit_template = load_module(sherlock_user_properties["CUSTOM_TRANSIT_TEMPLATE"])
        class_name = Path(sherlock_user_properties["CUSTOM_TRANSIT_TEMPLATE"].replace(".py", "")).name
        user_transit_template = getattr(user_transit_template, class_name)()
        globals()[class_name] = user_transit_template
        pickle.dumps(user_transit_template)
    ## Adding all object infos to same array
    object_infos.extend(mission_object_infos)
    object_infos.extend(ffi_object_infos)
    object_infos.extend(ffi_coords_object_infos)
    object_infos.extend(input_object_infos)
    object_infos.extend(input_id_object_infos)
    sherlock.Sherlock(sherlock_user_properties["UPDATE_OIS"], object_infos, args.explore) \
        .setup_detrend(sherlock_user_properties["INITIAL_SMOOTH_ENABLED"],
                       sherlock_user_properties["INITIAL_HIGH_RMS_MASK"],
                       sherlock_user_properties["INITIAL_HIGH_RMS_THRESHOLD"],
                       sherlock_user_properties["INITIAL_HIGH_RMS_BIN_HOURS"],
                       sherlock_user_properties["DETRENDS_NUMBER"],
                       sherlock_user_properties["DETREND_METHOD"], sherlock_user_properties["DETREND_L_MIN"],
                       sherlock_user_properties["DETREND_L_MAX"], sherlock_user_properties["DETREND_CORES"],
                       sherlock_user_properties["AUTO_DETREND_ENABLED"], sherlock_user_properties["AUTO_DETREND_RATIO"],
                       sherlock_user_properties["AUTO_DETREND_METHOD"],
                       user_prepare) \
        .setup_transit_adjust_params(sherlock_user_properties["MAX_RUNS"], sherlock_user_properties["MIN_SECTORS"],
                                     sherlock_user_properties["MAX_SECTORS"],
                                     sherlock_user_properties["PERIOD_PROTECT"],
                                     sherlock_user_properties["SEARCH_ZONE"], user_search_zone,
                                     sherlock_user_properties["PERIOD_MIN"], sherlock_user_properties["PERIOD_MAX"],
                                     sherlock_user_properties["BIN_MINUTES"], sherlock_user_properties["CPU_CORES"],
                                     sherlock_user_properties["SNR_MIN"], sherlock_user_properties["SDE_MIN"],
                                     sherlock_user_properties["FAP_MAX"], sherlock_user_properties["MASK_MODE"],
                                     sherlock_user_properties["BEST_SIGNAL_ALGORITHM"],
                                     sherlock_user_properties["QUORUM_STRENGTH"],
                                     sherlock_user_properties["MIN_QUORUM"],
                                     user_selection_algorithm,
                                     sherlock_user_properties["FIT_METHOD"],
                                     sherlock_user_properties["OVERSAMPLING"],
                                     sherlock_user_properties["T0_FIT_MARGIN"],
                                     sherlock_user_properties["DURATION_GRID_STEP"],
                                     user_transit_template) \
        .run()
