import numpy
import pandas
import pandas as pd
from astroquery.mast import Catalogs
from astroquery.mast import Observations


class MastCatalog:
    """
    Utility for querying the MAST (Mikulski Archive for Space Telescopes) catalog
    to cross-match Gaia DR3 IDs with TIC IDs and retrieve TESS observation data.
    """

    @staticmethod
    def gaia_to_tic(gaia_ids: list) -> pd.DataFrame:
        """
        Cross-match a list of Gaia DR3 IDs to TIC IDs and retrieve TESS sector coverage.

        Queries the MAST TIC catalog for each Gaia ID, retrieves TIC identifiers
        and Tmag values, then queries the MAST Observations database for TESS
        time-series coverage of each target.

        Parameters
        ----------
        gaia_ids : list of int
            List of Gaia DR3 source IDs.

        Returns
        -------
        stars_df : pandas.DataFrame
            DataFrame with columns 'Gaia_ID', 'TIC_ID', 'Tmag', and 'TIC_sectors'.
        duplicated_df : pandas.DataFrame
            DataFrame of duplicate Gaia-TIC crossmatches.
        """
        stars_df = pd.DataFrame(columns=['Gaia_ID', 'TIC_ID', 'Tmag'])
        stars_df['Gaia_ID'] = gaia_ids
        stars_df['TIC_ID'] = ''
        stars_df['Tmag'] = numpy.nan
        catalog_data = Catalogs.query_criteria(GAIA=gaia_ids, catalog="Tic", objType="STAR")
        catalog_data = catalog_data['GAIA', 'ID', 'Tmag']
        catalog_data_df = catalog_data.to_pandas()
        duplicated_1 = catalog_data_df.loc[catalog_data_df['GAIA'].duplicated(keep='first'), ['GAIA', 'ID', 'Tmag']]
        duplicated_2 = catalog_data_df.loc[catalog_data_df['GAIA'].duplicated(keep='last'), ['GAIA', 'ID', 'Tmag']]
        duplicated_df = pandas.concat([duplicated_1, duplicated_2])
        for catalog_data_row in catalog_data:
            stars_df.loc[stars_df['Gaia_ID'] == int(catalog_data_row['GAIA']), 'TIC_ID'] = catalog_data_row['ID']
            stars_df.loc[stars_df['Gaia_ID'] == int(catalog_data_row['GAIA']), 'Tmag'] = catalog_data_row['Tmag']
        stars_df.loc[stars_df['Gaia_ID'] == int(catalog_data_row['GAIA']), 'Tmag'] = catalog_data_row['Tmag']
        observations = Observations.query_criteria(obs_collection='*',
                                                   target_name=stars_df.loc[:, 'TIC_ID'].to_numpy(),
                                                   dataproduct_type=['timeseries', 'image'])
        observations_df = observations.to_pandas()
        observations_df.sort_values(by=['target_name', 'sequence_number'], inplace=True)
        stars_df['TIC_sectors'] = ''
        for index, star_df_row in stars_df.iterrows():
            observations_rows = observations_df.loc[
                observations_df['target_name'] == star_df_row['TIC_ID'], 'sequence_number'].drop_duplicates()
            if len(observations_rows) > 0:
                stars_df.loc[stars_df['TIC_ID'] == star_df_row['TIC_ID'], 'TIC_sectors'] = '[' + ','.join(
                    map(str, observations_rows)) + ']'
        stars_df.to_csv("/home/martin/Downloads/status_2023-9-27_tic.csv")
        duplicated_df.sort_values(by=['GAIA', 'ID'], ascending=True, inplace=True)
        return stars_df, duplicated_df

stars_df, duplicated_df = MastCatalog.gaia_to_tic([1762524526668814208,
419622462350748928,
423026756506669952,
419622977746938880,
513683006345633408,
513876859692937600,
3257254994871286400,
229154033007516800,
229154410964595328,
3421839717904456576,
279844229168337152,
279844263528063360,
3015582437867778048,
191298049523508608,
4656781839316354816,
3005434362156520704,
3005422611125997568,
2945644534751752704,
5565108770335311360,
3048537073399263360,
3135477931704572800,
1085034465548700800,
703779392734147584,
703978507417193344,
763969850397037952,
6057069339664907776,
5872182405008556288,
6225138041345032448,
1272190173733760128,
5985286073798439040,
5889224216794740736,
5809378021612555008,
4327593349348911360,
6636966061468135040])
print(stars_df.to_string())
