import sys

import pandas
from astropy.table import unique
from astropy.utils.diff import report_diff_values
from astroquery.mast import Catalogs
from astroquery.mast import Observations


class MastCatalog:
    @staticmethod
    def gaia_to_tic(gaia_ids: list):
        catalog_data = Catalogs.query_criteria(GAIA=gaia_ids, catalog="Tic",
                                               objType="STAR")
        catalog_data = catalog_data['GAIA', 'ID']
        catalog_data_df = catalog_data.to_pandas()
        duplicated_1 = catalog_data_df.loc[catalog_data_df['GAIA'].duplicated(keep='first'), ['GAIA', 'ID']]
        duplicated_2 = catalog_data_df.loc[catalog_data_df['GAIA'].duplicated(keep='last'), ['GAIA', 'ID']]
        duplicated_df = pandas.concat([duplicated_1, duplicated_2])
        duplicated_df.sort_values(by=['GAIA', 'ID'], ascending=True, inplace=True)
        for catalog_data_row in catalog_data:
            stars_df.loc[stars_df['Gaia_ID'] == int(catalog_data_row['GAIA']), 'TIC_ID'] = catalog_data_row['ID']


stars_df = pandas.read_csv("/home/martin/Downloads/status_2023-9-27.csv")
stars_df['TIC_ID'] = ''
catalog_data = Catalogs.query_criteria(GAIA=stars_df.loc[:, 'Gaia_ID'].to_numpy(), catalog="Tic", objType="STAR")
catalog_data = catalog_data['GAIA', 'ID']
catalog_data_df = catalog_data.to_pandas()
duplicated_1 = catalog_data_df.loc[catalog_data_df['GAIA'].duplicated(keep='first'), ['GAIA', 'ID']]
duplicated_2 = catalog_data_df.loc[catalog_data_df['GAIA'].duplicated(keep='last'), ['GAIA', 'ID']]
duplicated_df = pandas.concat([duplicated_1, duplicated_2])
for catalog_data_row in catalog_data:
    stars_df.loc[stars_df['Gaia_ID'] == int(catalog_data_row['GAIA']), 'TIC_ID'] = catalog_data_row['ID']
observations = Observations.query_criteria(obs_collection='TESS', target_name=stars_df.loc[:, 'TIC_ID'].to_numpy())
observations_df = observations.to_pandas()
observations_df.sort_values(by=['target_name', 'sequence_number'], inplace=True)
stars_df['TIC_sectors'] = ''
for index, star_df_row in stars_df.iterrows():
    observations_rows = observations_df.loc[observations_df['target_name'] == star_df_row['TIC_ID'], 'sequence_number'].drop_duplicates()
    if len(observations_rows) > 0:
        stars_df.loc[stars_df['TIC_ID'] == star_df_row['TIC_ID'], 'TIC_sectors'] = '[' + ','.join(map(str, observations_rows)) + ']'
stars_df.to_csv("/home/martin/Downloads/status_2023-9-27_tic.csv")
duplicated_df.sort_values(by=['GAIA', 'ID'], ascending=True, inplace=True)
duplicated_df.to_csv("/home/martin/Downloads/status_2023-9-27_duplicated_tics.csv")
