import os
from pathlib import Path
import pandas as pd
import requests


class OisManager:
    """
    Handles the tois, kois and epic ois for the different missions. Updates them from the proper web services.
    """
    TOIS_CSV_URL = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
    CTOIS_CSV_URL = 'https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv'
    KOIS_LIST_URL = 'https://exofop.ipac.caltech.edu/kepler/targets.php?sort=num-pc&page1=1&ipp1=100000&koi1=&koi2='
    KIC_STAR_URL = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=keplerstellar&select=kepid,dist'
    KOI_CSV_URL = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative'
    EPIC_CSV_URL = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+epic_hostname,pl_name,' \
                   'disposition,pl_orbper,sy_dist,st_teff,st_logg,st_metratio,st_metratio,st_vsin,sy_kepmag,' \
                   'pl_trandep,pl_trandur,pl_rade,pl_eqt,pl_orbincl,ra,dec,pl_tranmid+from+k2pandc&format=csv'
    TOIS_CSV = 'tois.csv'
    CTOIS_CSV = 'ctois.csv'
    KOIS_CSV = 'kois.csv'
    EPIC_CSV = 'epic_ois.csv'
    KIC_STAR_CSV = 'kic_star.csv'
    ois = None

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        home = self.cache_dir + "/.sherlockpipe/"
        if not os.path.exists(home):
            os.mkdir(home)
        self.tois_csv = home + self.TOIS_CSV
        self.ctois_csv = home + self.CTOIS_CSV
        self.kois_csv = home + self.KOIS_CSV
        self.epic_csv = home + self.EPIC_CSV
        self.kic_star_csv = home + self.KIC_STAR_CSV

    def load_ois(self):
        """
        Loads all the ois in an in-memory dataframe.
        @return: the ois dataframe
        """
        if not os.path.isfile(self.tois_csv) or not os.path.isfile(self.ctois_csv):
            print("TOIs files are not found. Downloading...")
            self.update_tic_csvs()
            print("TOIs files download is finished!")
        toi_data = pd.read_csv(self.tois_csv)
        ois = toi_data
        ctoi_data = pd.read_csv(self.ctois_csv)
        ois = pd.concat([ois, ctoi_data])
        if not os.path.isfile(self.kois_csv):
            print("KOIs files are not found. Downloading...")
            self.update_kic_csvs()
            print("KOIs files download is finished!")
        koi_data = pd.read_csv(self.kois_csv)
        ois = pd.concat([ois, koi_data])
        if not os.path.isfile(self.epic_csv):
            print("EPIC IDs files are not found. Downloading...")
            self.update_epic_csvs()
            print("EPIC IDs files download is finished!")
        epic_data = pd.read_csv(self.epic_csv)
        ois = pd.concat([ois, epic_data])
        return ois

    def update_tic_csvs(self):
        """
        Reloads the TESS Objects of Interest.
        @return: the OisManager class to be used as a fluent API.
        """
        tic_csv = open(self.tois_csv, 'wb')
        request = requests.get(self.TOIS_CSV_URL)
        tic_csv.write(request.content)
        tic_csv.close()
        tic_csv = open(self.ctois_csv, 'wb')
        request = requests.get(self.CTOIS_CSV_URL)
        tic_csv.write(request.content)
        tic_csv.close()
        toi_data = pd.read_csv(self.tois_csv)
        toi_data['TIC ID'] = 'TIC ' + toi_data['TIC ID'].astype(str)
        toi_data['TOI'] = 'TOI ' + toi_data['TOI'].astype(str)
        toi_data = toi_data.rename(columns={'TOI': 'OI', 'TIC ID': 'Object Id', 'TFOPWG Disposition': 'Disposition'})
        toi_data.to_csv(self.tois_csv, index=False)
        ctoi_data = pd.read_csv(self.ctois_csv)
        ctoi_data['TIC ID'] = 'TIC ' + ctoi_data['TIC ID'].astype(str)
        ctoi_data['CTOI'] = 'CTOI ' + ctoi_data['CTOI'].astype(str)
        ctoi_data = ctoi_data.rename(columns={'CTOI': 'OI', 'TIC ID': 'Object Id', 'TFOPWG Disposition': 'Disposition',
                                 'Duration (hrs)': 'Duration (hours)', 'Duration (hrs) Err': 'Duration (hours) err',
                                 'Period (days) Err': 'Period (days) err', 'Radius (R_Earth)': 'Planet Radius (R_Earth)',
                                 'Depth mmag': 'Depth (mmag)', 'Depth mmag Err': 'Depth (mmag) err',
                                 'Depth ppm': 'Depth (ppm)', 'Depth ppm Err': 'Depth (ppm) err',
                                 'Insolation (Earth Flux)': 'Planet Insolation (Earth Flux)',
                                 'Equilibrium Temp (K)': 'Planet Equil Temp (K)'})
        ctoi_data.to_csv(self.ctois_csv, index=False)
        return self

    def update_kic_csvs(self):
        """
        Reloads the Kepler Objects of Interest.
        @return: the OisManager class to be used as a fluent API.
        """
        koi_csv = open(self.kois_csv, 'wb')
        request = requests.get(self.KOI_CSV_URL)
        koi_csv.write(request.content)
        koi_csv.close()
        kic_star_csv = open(self.kic_star_csv, 'wb')
        request = requests.get(self.KIC_STAR_URL)
        kic_star_csv.write(request.content)
        kic_star_csv.close()
        koi_data = pd.read_csv(self.kois_csv)
        koi_data['kepid'] = 'KIC ' + koi_data['kepid'].astype(str)
        koi_data['ra_str'] = koi_data['ra_str'].astype(str).replace("h", ":").replace("m", ":").replace("s", "")
        koi_data['dec_str'] = koi_data['dec_str'].astype(str).replace("d", ":").replace("m", ":").replace("s", "")
        koi_data['koi_time0bk'] = koi_data['koi_time0bk'].astype(float) + 2454833.0
        kic_star_data = pd.read_csv(self.kic_star_csv)
        kic_star_data['dist'] = kic_star_data['dist'].astype(float)
        kic_star_data['kepid'] = 'KIC ' + kic_star_data['kepid'].astype(str)
        kic_star_data = kic_star_data[kic_star_data['dist'].notnull()]
        kic_star_data = kic_star_data.groupby(['kepid'])['dist'].mean()
        koi_data = pd.merge(koi_data, kic_star_data, on='kepid', how='left')
        koi_data = koi_data.rename(columns={'kepoi_name': 'OI', 'kepid': 'Object Id',
                                            'koi_disposition': 'Disposition', 'koi_duration': 'Duration (hours)',
                                            'koi_duration_err1': 'Duration (hours) err',
                                            'koi_period': 'Period (days) err', 'koi_period_err1': 'Period (days) err',
                                            'koi_time0bk': 'Epoch (BJD)', 'koi_time0bk_err1': 'Epoch (BJD) err',
                                            'koi_depth': 'Depth (ppm)', 'koi_depth_err1': 'Depth (ppm) err',
                                            'koi_insol': 'Planet Insolation (Earth Flux)',
                                            'koi_insol_err1': 'Planet Insolation (Earth Flux) err',
                                            'koi_teq': 'Planet Equil Temp (K)',
                                            'koi_teq_err1': 'Planet Equil Temp (K) err',
                                            'ra_str': 'RA', 'dec_str': 'Dec',
                                            'koi_slogg': 'Stellar log(g) (cm/s^2)',
                                            'koi_slogg_err1': 'Stellar log(g) (cm/s^2) err',
                                            'koi_steff': 'Stellar Eff Temp (K)',
                                            'koi_steff_err1': 'Stellar Eff Temp (K) err',
                                            'koi_prad': 'Planet Radius (R_Earth)',
                                            'koi_kepmag': 'Kepler Mag',
                                            'dist': 'Stellar Distance (pc)'})
        koi_data.to_csv(self.kois_csv, index=False)
        return self

    def update_epic_csvs(self):
        """
        Reloads the K2 Objects of Interest.
        @return: the OisManager class to be used as a fluent API.
        """
        epic_csv = open(self.epic_csv, 'wb')
        request = requests.get(self.EPIC_CSV_URL)
        epic_csv.write(request.content)
        epic_csv.close()
        epic_data = pd.read_csv(self.epic_csv)
        epic_data['pl_trandep'] = epic_data['pl_trandep'].astype(float) * 1000000
        epic_data['pl_tranmid'] = epic_data['pl_tranmid'].astype(float) - epic_data['pl_trandur'].astype(float) / 2
        epic_data['ra'] = epic_data['ra'].astype(str).replace("h", ":").replace("m", ":").replace("s", "")
        epic_data['dec'] = epic_data['dec'].astype(str).replace("d", ":").replace("m", ":").replace("s", "")
        epic_data[epic_data['disposition'] == "CANDIDATE"]['disposition'] = "PC"
        epic_data[epic_data['disposition'] == "CONFIRMED"]['disposition'] = "CP"
        epic_data[epic_data['disposition'] == "FALSE POSITIVE"]['disposition'] = "FP"
        epic_data = epic_data.rename(columns={'pl_name': 'OI', 'epic_hosname': 'Object Id',
                                            'disposition': 'Disposition', 'pl_trandur': 'Duration (hours)',
                                            'pl_orbper': 'Period (days)',
                                            'pl_tranmid': 'Epoch (BJD)',
                                            'pl_trandep': 'Depth (ppm)',
                                            'pl_eqt': 'Planet Equil Temp (K)',
                                            'ra': 'RA', 'dec': 'Dec',
                                            'st_logg': 'Stellar log(g) (cm/s^2)',
                                            'st_teff': 'Stellar Eff Temp (K)',
                                            'pl_rade': 'Planet Radius (R_Earth)',
                                            'sy_dist': 'Stellar Distance (pc)',
                                            'st_metratio': 'Stellar Metallicity'})
        epic_data.to_csv(self.epic_csv, index=False)
        return self
