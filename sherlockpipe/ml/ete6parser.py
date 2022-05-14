import gzip
import os
import shutil

import pandas as pd
import numpy as np
from lcbuilder.star.TicStarCatalog import TicStarCatalog
from lightkurve import TessLightCurve


def create_injection_dataframe(injections_dir, lcs_dir):
    backeb_df = pd.read_csv(injections_dir + '/ete6_backeb_data.txt', comment="#")
    eb_df = pd.read_csv(injections_dir + '/ete6_eb_data.txt', comment="#")
    planet_df = pd.read_csv(injections_dir + '/ete6_planet_data.txt', comment="#")
    df = pd.DataFrame(columns=['TIC ID', 'type', 'period', 'epoch', 'Rsec/Rpri', 'b',
                                'a/Rstar', 'duration(h)', 'depth_primary', 'depth_secondary',
                               'insolation', 'Rstar_primary', 'Rstar_secondary', 'contact_amplitude', 'Mstar'])
    i = 0
    log_count = 100
    for file in os.listdir(lcs_dir):
        lc_file = lcs_dir + '/' + file
        lc = TessLightCurve.read(lc_file)
        object_id = lc.meta['OBJECT']
        object_id = int(object_id.split(' ')[1])
        target_rows = backeb_df[backeb_df['TIC ID'] == object_id]
        for index, target_row in target_rows.iterrows():
            df.append({'TIC ID': target_row['TIC ID'], 'type': 'bckEB', 'period': target_row['Orbital Period'],
                       'epoch': target_row['Epoch [BTJD]'],
                       'Rsec/Rpri': target_row['Secondary Rstar'] / target_row['Primary Rstar'],
                       'b': target_row['Impact Parameter'],
                       'a/Rstar': np.nan, 'duration(h)': np.nan,
                       'depth_primary': target_row['Primary Star Depth'],
                       'depth_secondary': target_row['Primary Star Depth'],
                       'insolation': np.nan, 'Rstar_primary': target_row['Primary Rstar'],
                       'Rstar_secondary': target_row['Secondary Rstar'],
                       'contact_amplitude': target_row['Contact Binary Amplitude'],
                       'Mstar': np.nan})
        target_rows = eb_df[eb_df['TIC ID'] == object_id]
        for index, target_row in target_rows.iterrows():
            df.append({'TIC ID': target_row['TIC ID'], 'type': 'EB', 'period': target_row['Orbital Period'],
                       'epoch': target_row['Epoch [BTJD]'],
                       'Rsec/Rpri': target_row['Secondary Rstar'] / target_row['Primary Rstar'],
                       'b': target_row['Impact Parameter'],
                       'a/Rstar': np.nan, 'duration(h)': np.nan,
                       'depth_primary': target_row['Primary Star Depth'],
                       'depth_secondary': target_row['Primary Star Depth'],
                       'insolation': np.nan, 'Rstar_primary': target_row['Primary Rstar'],
                       'Rstar_secondary': target_row['Secondary Rstar'],
                       'contact_amplitude': target_row['Contact Binary Amplitude'],
                       'Mstar': np.nan})
        target_rows = planet_df[planet_df['TIC ID'] == object_id]
        for index, target_row in target_rows.iterrows():
            df.append({'TIC ID': target_row['TIC ID'], 'type': 'planet', 'period': target_row['Orbital Period'],
                       'epoch': target_row['Epoch [BTJD]'],
                       'Rsec/Rpri': target_row['Rp/Rstar'],
                       'b': target_row['Impact Parameter'],
                       'a/Rstar': target_row['a/Rstar'], 'duration(h)': target_row['duration[h]'],
                       'depth_primary': target_row['depth'],
                       'depth_secondary': np.nan,
                       'insolation': target_row['Insolation Flux'],
                       'Rstar_primary': target_row['Rstar'],
                       'Rstar_secondary': target_row['Rstar'] * target_row['Rp/Rstar'],
                       'contact_amplitude': np.nan,
                       'Mstar': target_row['Mstar']})
        df = df.sort_values(["TIC ID", "type"], ascending=True)
        df.to_csv(injections_dir + '/injected_objects.csv')
        if i % log_count == 0:
            print("Processed " + str(i) + " TICs. Found ")


def create_target_csvs(lcs_dir, models_dir, max_lc_length=20610):
    for file in os.listdir(lcs_dir):
        lc_file = lcs_dir + '/' + file
        lc_df = pd.DataFrame(columns=['#time', 'flux', 'flux_err', 'planet_model', 'eb_model', 'bckeb_model',
                                      'centroid_x', 'centroid_y', 'motion_x', 'motion_y', 'bck_flux'])
        lc = TessLightCurve.read(lc_file)
        object_id = lc.meta['OBJECT']
        object_id = int(object_id.split(' ')[1])
        leading_zeros_object_id = '{:09}'.format(object_id)
        lc_length = len(lc)
        eb_model_flux = np.ones(lc_length)
        backeb_model_flux = np.ones(lc_length)
        planet_model_flux = np.ones(lc_length)
        model_file = models_dir + '/ebs/EBs/EBs' + '_' + leading_zeros_object_id + '.txt'
        if os.path.exists(model_file):
            model_df = pd.read_csv(model_file, header=None)
            eb_model_flux = model_df[0].to_numpy()
            eb_model_flux = eb_model_flux / np.median(eb_model_flux)
            eb_model_flux = resample_model(eb_model_flux, lc.time.value)
        model_file = models_dir + '/backebs/BackEBs/BackEBs_' + '_' + leading_zeros_object_id + '.txt'
        if os.path.exists(model_file):
            model_df = pd.read_csv(model_file, header=None)
            backeb_model_flux = model_df[0].to_numpy()
            backeb_model_flux = backeb_model_flux / np.median(backeb_model_flux)
            resample_model(backeb_model_flux, lc.time.value)
            # import matplotlib.pyplot as plt
            # plt.scatter(np.linspace(lc.time.value[0], lc.time.value[-1], len(eb_model_flux)), eb_model_flux)
            # plt.scatter(lc.time.value, lc.pdcsap_flux.value / np.median(lc.pdcsap_flux.value))
            # plt.show()
        model_file = models_dir + '/planets/Planets/Planets' + '_' + leading_zeros_object_id + '.txt'
        if os.path.exists(model_file):
            model_df = pd.read_csv(model_file, header=None)
            planet_model_flux = model_df[0].to_numpy()
            planet_model_flux = planet_model_flux / np.median(planet_model_flux)
            resample_model(planet_model_flux, lc.time.value)
        padding_zeros_count = max_lc_length - lc_length
        padding_zeros_count = 0 if padding_zeros_count < 0 else padding_zeros_count
        padding_zeros = np.zeros(padding_zeros_count)
        padding_ones = np.ones(padding_zeros_count)
        padding_nans = np.full(padding_zeros_count, np.nan)
        lc_df['#time'] = np.concatenate((np.array(lc.time.value), padding_nans))
        lc_df['flux'] = np.concatenate((np.array(lc.pdcsap_flux.value), padding_ones))
        lc_df['flux_err'] = np.concatenate((np.array(lc.time.value), padding_zeros))
        lc_df['centroid_x'] = np.concatenate((np.array(lc.centroid_col.value), padding_nans))
        lc_df['centroid_y'] = np.concatenate((np.array(lc.centroid_row.value), padding_nans))
        lc_df['motion_x'] = np.concatenate((np.array(lc.mom_centr1.value), padding_nans))
        lc_df['motion_y'] = np.concatenate((np.array(lc.mom_centr2.value), padding_nans))
        lc_df['bck_flux'] = np.concatenate((np.array(lc.sap_bkg.value), padding_nans))
        lc_df['eb_model'] = np.concatenate((np.array(eb_model_flux), padding_ones))
        lc_df['bckeb_model'] = np.concatenate((np.array(backeb_model_flux), padding_ones))
        lc_df['planet_model'] = np.concatenate((np.array(planet_model_flux), padding_ones))
        lc_df.to_csv(lcs_dir + '/' + leading_zeros_object_id + '_lc.csv')
        # star_info = TicStarCatalog().catalog_info(object_id)
        # star_df = pd.DataFrame(columns=['ld_a', 'ld_b', 'Teff', 'lum', 'logg', 'radius', 'mass', 'v', 'j', 'h', 'k'])
        # star_df.append({'ld_a': star_info[0][0], 'ld_b': star_info[0][1], 'Teff': star_info[1], 'lum': star_info[2],
        #                 'logg': star_info[3], 'radius': star_info[5], 'mass': star_info[8], 'v': star_info[13],
        #                 'j': star_info[15], 'h': star_info[17], 'k': star_info[19]}, ignore_index=True)
        # star_df.to_csv(lcs_dir + '/' + leading_zeros_object_id + '_star.csv')

def resample_model(model, lc_time):
    total_missing_points = len(model) - len(lc_time)
    gaps_sizes = []
    time_gaps_indexes = np.argwhere(np.abs(lc_time[1:] - lc_time[:-1]) > 0.014).flatten()
    for time_gap_index in time_gaps_indexes:
        gaps_sizes = gaps_sizes + [lc_time[time_gap_index + 1] - lc_time[time_gap_index]]
    total_gap_size = np.sum(gaps_sizes)
    gap_sizes_percent = [gap_size / total_gap_size for gap_size in gaps_sizes]
    gap_points = []
    for gap_size_percent in gap_sizes_percent[:-1]:
        gap_points = gap_points + [int(np.round(total_missing_points * gap_size_percent))]
    gap_points = gap_points + [total_missing_points - np.sum(gap_points)]
    added_gap_points = 0
    i = 0
    corrected_time_gap_indexes = np.array([])
    for time_gap_index in time_gaps_indexes:
        corrected_time_gap_index = time_gap_index + added_gap_points
        corrected_time_gap_indexes = np.concatenate((corrected_time_gap_indexes, np.arange(corrected_time_gap_index + 1, corrected_time_gap_index + 1 + gap_points[i])))
        lc_time = np.concatenate((lc_time[0:corrected_time_gap_index + 1],
                                  np.linspace(lc_time[corrected_time_gap_index] + 0.001388, lc_time[corrected_time_gap_index + 1], gap_points[i]).flatten(),
                                  lc_time[corrected_time_gap_index + 1:]))
        added_gap_points = gap_points[i]
        i = i + 1
    model = np.delete(model, corrected_time_gap_indexes.astype(int))

    # time_gaps_indexes = np.argwhere(np.abs(lc_time[1:] - lc_time[:-1]) > 0.014).flatten()
    # time_gap_range_bottom = lc_time[time_gaps_indexes].flatten()
    # time_gap_range_up = lc_time[time_gaps_indexes + 1].flatten()
    # time_model = np.linspace(lc_time[0], lc_time[-1], len(model))
    # time_values_to_add = []
    # i = 0
    # for time_gap_index in time_gaps_indexes:
    #     time_values_to_add = np.concatenate((time_values_to_add, np.arange(lc_time[time_gap_index] + 0.00138888,
    #                                                                        lc_time[time_gap_index + 1], 0.00138888)))
    #     i = i + 1
    return model

def uncompress_data(data_dir):
    for file in os.listdir(data_dir):
        original_file_name = data_dir + '/' + file
        destination_file_name = file.split('.')[0]
        destination_file_name = data_dir + '/' + destination_file_name + '.txt'
        with gzip.open(original_file_name, 'rb') as f_in:
            with open(destination_file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


root_dir = '/mnt/DATA-2/ete6/'
lcs_dir = root_dir + 'lcs/'
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')
#create_injection_dataframe(root_dir, lcs_dir)
create_target_csvs(lcs_dir, root_dir)
