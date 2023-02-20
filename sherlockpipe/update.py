import logging
import os
import shutil
import time
import traceback
from argparse import ArgumentParser
from lcbuilder.constants import USER_HOME_ELEANOR_CACHE
from lcbuilder.eleanor_manager import EleanorManager
from sherlockpipe.ois.OisManager import OisManager
from eleanor.maxsector import maxsector


class Updater:
    """
    This class should be used to download SHERLOCK data as objects of interest, ELEANOR or LATTE metadata. It stores
    timestamps to remind when was the last time it updated and if called again, it could do nothing if it reads very
    recent operations.
    """
    def __init__(self, cache_dir=os.path.expanduser('~') + "/"):
        self.cache_dir = cache_dir

    def update(self, clean, ois, force):
        """
        The main method of the update task. Reloads OIs, ELEANOR and LATTE metadata.

        :param clean: Specifies whether the ELEANOR data should be wiped and downloaded again.
        :param ois: Specifies whether the OIs metadata is the only one to be refreshed (ignoring ELEANOR and LATTE)
        :param force: Specifies whether the last download timestamp should be ignored and proceed as if a refresh was needed.
        """
        ois_manager = OisManager(self.cache_dir)
        timestamp_ois_path = os.path.join(self.cache_dir, '.sherlockpipe/timestamp_ois.txt')
        ois_timestamp = 0
        force = force or clean
        if os.path.exists(timestamp_ois_path):
            with open(timestamp_ois_path, 'r+') as f:
                ois_timestamp = f.read()
        if force or time.time() - float(ois_timestamp) > 3600 * 24 * 7:
            logging.info("------------------ Reloading TOIs ------------------")
            ois_manager.update_tic_csvs()
            logging.info("------------------ Reloading KOIs ------------------")
            ois_manager.update_kic_csvs()
            logging.info("------------------ Reloading EPICs ------------------")
            ois_manager.update_epic_csvs()
            with open(os.path.join(os.path.expanduser('~'), '.sherlockpipe/timestamp_ois.txt'), 'w+') as f:
                f.write(str(time.time()))
        print("DONE")


if __name__ == '__main__':
    ap = ArgumentParser(description='Updater of SHERLOCK PIPEline metadata')
    ap.add_argument('--clean', dest='clean', action='store_true', help="Whether to remove all data and download it again.")
    ap.add_argument('--only_ois', dest='ois', action='store_true', help="Whether to only refresh objects of interest.")
    ap.add_argument('--force', dest='force', action='store_true', help="Whether to ignore update timestamps and do everything again.")
    args = ap.parse_args()
    Updater(USER_HOME_ELEANOR_CACHE).update(args.clean, args.ois, args.force)
