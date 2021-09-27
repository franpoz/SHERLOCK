import os
import shutil
import sys
import time
import traceback
from argparse import ArgumentParser
import lcbuilder.eleanor
sys.modules['eleanor'] = sys.modules['lcbuilder.eleanor']
import eleanor
from sherlockpipe.vet import Vetter
from sherlockpipe.ois.OisManager import OisManager
from eleanor.maxsector import maxsector


class Updater:
    """
    This class should be used to download SHERLOCK data as objects of interest, ELEANOR or LATTE metadata. It stores
    timestamps to remind when was the last time it updated and if called again, it could do nothing if it reads very
    recent operations.
    """
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def update(self, clean, ois, force):
        """
        The main method of the update task. Reloads OIs, ELEANOR and LATTE metadata.
        @param clean: Specifies whether the ELEANOR data should be wiped and downloaded again.
        @param ois: Specifies whether the OIs metadata is the only one to be refreshed (ignoring ELEANOR and LATTE)
        @param force: Specifies whether the last download timestamp should be ignored and proceed as if a refresh was
        needed.
        """
        ois_manager = OisManager(self.cache_dir)
        timestamp_ois_path = os.path.join(self.cache_dir, '.sherlockpipe/timestamp_ois.txt')
        timestamp_eleanor_path = os.path.join(self.cache_dir, '.sherlockpipe/timestamp_eleanor.txt')
        timestamp_latte_path = os.path.join(self.cache_dir, '.sherlockpipe/timestamp_latte.txt')
        ois_timestamp = 0
        eleanor_timestamp = 0
        latte_timestamp = 0
        force = force or clean
        if os.path.exists(timestamp_ois_path):
            with open(timestamp_ois_path, 'r+') as f:
                ois_timestamp = f.read()
        if os.path.exists(timestamp_eleanor_path):
            with open(timestamp_eleanor_path, 'r+') as f:
                eleanor_timestamp = f.read()
        if os.path.exists(timestamp_latte_path):
            with open(timestamp_latte_path, 'r+') as f:
                latte_timestamp = f.read()
        if force or time.time() - float(ois_timestamp) > 3600 * 24 * 7:
            print("------------------ Reloading TOIs ------------------")
            ois_manager.update_tic_csvs()
            print("------------------ Reloading KOIs ------------------")
            ois_manager.update_kic_csvs()
            print("------------------ Reloading EPICs ------------------")
            ois_manager.update_epic_csvs()
            with open(os.path.join(os.path.expanduser('~'), '.sherlockpipe/timestamp_ois.txt'), 'w+') as f:
                f.write(str(time.time()))
        if force or time.time() - float(eleanor_timestamp) > 3600 * 24 * 7:
            print("------------------ Reloading ELEANOR TESS FFI data ------------------")
            eleanorpath = os.path.join(self.cache_dir, '.eleanor')
            eleanormetadata = eleanorpath + "/metadata"
            if clean and os.path.exists(eleanorpath) and os.path.exists(eleanormetadata):
                shutil.rmtree(eleanormetadata, ignore_errors=True)
            if not os.path.exists(eleanorpath):
                os.mkdir(eleanorpath)
            if not os.path.exists(eleanormetadata):
                os.mkdir(eleanormetadata)
            for sector in range(1, 52):
                sectorpath = eleanorpath + '/metadata/s{:04d}'.format(sector)
                if os.path.exists(sectorpath) and os.path.isdir(sectorpath) and not os.listdir(sectorpath):
                    os.rmdir(sectorpath)
                if (not os.path.exists(sectorpath) or not os.path.isdir(sectorpath) or not os.listdir(sectorpath)) and sector <= maxsector:
                    try:
                        eleanor.Update(sector)
                    except Exception as e:
                        traceback.print_exc()
                        shutil.rmtree(sectorpath)
                        break
            with open(os.path.join(os.path.expanduser('~'), '.sherlockpipe/timestamp_eleanor.txt'), 'w+') as f:
                f.write(str(time.time()))
        if (force or time.time() - float(latte_timestamp) > 3600 * 24 * 7) and not ois:
            print("------------------ Reloading LATTE data ------------------")
            Vetter(None, False).update()
            with open(os.path.join(os.path.expanduser('~'), '.sherlockpipe/timestamp_latte.txt'), 'w+') as f:
                f.write(str(time.time()))
        print("DONE")


if __name__ == '__main__':
    ap = ArgumentParser(description='Updater of SHERLOCK PIPEline metadata')
    ap.add_argument('--clean', dest='clean', action='store_true', help="Whether to remove all data and download it again.")
    ap.add_argument('--only_ois', dest='ois', action='store_true', help="Whether to only refresh objects of interest.")
    ap.add_argument('--force', dest='force', action='store_true', help="Whether to ignore update timestamps and do everything again.")
    args = ap.parse_args()
    Updater().update(args.clean, args.ois, args.force)
