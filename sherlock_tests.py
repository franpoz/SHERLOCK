import os
import shutil
import unittest

from objectinfo.InputObjectInfo import InputObjectInfo
from objectinfo.MissionFfiCoordsObjectInfo import MissionFfiCoordsObjectInfo
from objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from objectinfo.MissionInputObjectInfo import MissionInputObjectInfo
from objectinfo.MissionObjectInfo import MissionObjectInfo
from objectinfo.preparer.MissionFfiLightcurveBuilder import MissionFfiLightcurveBuilder
from objectinfo.preparer.MissionInputLightcurveBuilder import MissionInputLightcurveBuilder
from objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder
from scoring.BasicSignalSelector import BasicSignalSelector
from scoring.QuorumSnrBorderCorrectedSignalSelector import QuorumSnrBorderCorrectedSignalSelector
from scoring.SnrBorderCorrectedSignalSelector import SnrBorderCorrectedSignalSelector

from sherlock_class import Sherlock


class SherlockTests(unittest.TestCase):
    def test_setup_files(self):
        sherlock = Sherlock(None)
        sherlock.setup_files("inner/")
        self.assertEquals("inner/", sherlock.results_dir)

    def test_setup_detrend(self):
        sherlock = Sherlock(None)
        sherlock.setup_detrend(initial_smooth=False, initial_rms_mask=False, initial_rms_threshold=3,
                               initial_rms_bin_hours=9, n_detrends=2, detrend_method="gp", cores=3,
                               auto_detrend_periodic_signals=True, auto_detrend_ratio=1 / 2,
                               auto_detrend_method="cosine")
        self.assertEquals(False, sherlock.initial_smooth)
        self.assertEquals(False, sherlock.initial_rms_mask)
        self.assertEquals(3, sherlock.initial_rms_threshold)
        self.assertEquals(9, sherlock.initial_rms_bin_hours)
        self.assertEquals(2, sherlock.n_detrends)
        self.assertEquals("gp", sherlock.detrend_method)
        self.assertEquals(3, sherlock.detrend_cores)
        self.assertEquals(True, sherlock.auto_detrend_periodic_signals)
        self.assertEquals(1 / 2, sherlock.auto_detrend_ratio)
        self.assertEquals("cosine", sherlock.auto_detrend_method)

    def test_setup_transit_adjust_params(self):
        sherlock = Sherlock(None)
        sherlock.setup_transit_adjust_params(max_runs=5, period_protec=5, period_min=1, period_max=2, bin_minutes=5,
                                             run_cores=3, snr_min=6, sde_min=5, fap_max=0.05, mask_mode="subtract",
                                             best_signal_algorithm="quorum", quorum_strength=2)
        self.assertEquals(5, sherlock.max_runs)
        self.assertEquals(5, sherlock.period_protec)
        self.assertEquals(1, sherlock.period_min)
        self.assertEquals(2, sherlock.period_max)
        self.assertEquals(5, sherlock.bin_minutes)
        self.assertEquals(3, sherlock.run_cores)
        self.assertEquals(6, sherlock.snr_min)
        self.assertEquals(5, sherlock.sde_min)
        self.assertEquals(0.05, sherlock.fap_max)
        self.assertEquals("subtract", sherlock.mask_mode)
        self.assertEquals("quorum", sherlock.best_signal_algorithm)
        # TODO test quorum strength

    def test_scoring_algorithm(self):
        sherlock = Sherlock(None)
        sherlock.setup_transit_adjust_params(best_signal_algorithm="basic")
        self.assertTrue(isinstance(sherlock.signal_score_selectors[sherlock.best_signal_algorithm],
                                   BasicSignalSelector))
        sherlock.setup_transit_adjust_params(best_signal_algorithm="border-correct")
        self.assertTrue(isinstance(sherlock.signal_score_selectors[sherlock.best_signal_algorithm],
                                   SnrBorderCorrectedSignalSelector))
        sherlock.setup_transit_adjust_params(best_signal_algorithm="quorum")
        self.assertTrue(isinstance(sherlock.signal_score_selectors[sherlock.best_signal_algorithm],
                                   QuorumSnrBorderCorrectedSignalSelector))

    def test_preparer(self):
        object_info = MissionObjectInfo("TIC 1234567", 'all')
        sherlock = Sherlock(object_info)
        self.assertTrue(isinstance(sherlock.lightcurve_builders[type(object_info)], MissionLightcurveBuilder))
        object_info = MissionFfiIdObjectInfo("TIC 1234567", 'all')
        sherlock = Sherlock(object_info)
        self.assertTrue(isinstance(sherlock.lightcurve_builders[type(object_info)], MissionFfiLightcurveBuilder))
        object_info = MissionFfiCoordsObjectInfo(19, 15, 'all')
        sherlock = Sherlock(object_info)
        self.assertTrue(isinstance(sherlock.lightcurve_builders[type(object_info)], MissionFfiLightcurveBuilder))
        object_info = MissionInputObjectInfo("TIC 1234567", "testfilename.csv")
        sherlock = Sherlock(object_info)
        self.assertTrue(isinstance(sherlock.lightcurve_builders[type(object_info)], MissionInputLightcurveBuilder))
        object_info = InputObjectInfo("testfilename.csv")
        sherlock = Sherlock(object_info)
        self.assertTrue(isinstance(sherlock.lightcurve_builders[type(object_info)], MissionInputLightcurveBuilder))

    def test_refresh_tois(self):
        sherlock = Sherlock(None)
        sherlock.ois_manager.update_tic_csvs()
        try:
            self.assertTrue(os.path.isfile("tois.csv"))
        finally:
            os.remove("tois.csv")

    def test_refresh_kois(self):
        sherlock = Sherlock(None)
        sherlock.ois_manager.update_kic_csvs()
        try:
            self.assertTrue(os.path.isfile("kic_star.csv"))
        finally:
            os.remove("kic_star.csv")
        try:
            self.assertTrue(os.path.isfile("kois.csv"))
        finally:
            os.remove("kois.csv")

    def test_refresh_epicois(self):
        sherlock = Sherlock(None)
        sherlock.ois_manager.update_epic_csvs()
        try:
            self.assertTrue(os.path.isfile("epic_ois.csv"))
        finally:
            os.remove("epic_ois.csv")

    def test_ois_loaded(self):
        sherlock = Sherlock(None)
        sherlock.refresh_ois()
        sherlock.load_ois()
        sherlock.filter_hj_ois()
        try:
            self.assertGreater(len(sherlock.ois.index), 100)
            sherlock.limit_ois(0, 5)
            self.assertEquals(len(sherlock.ois.index), 5)
            self.assertTrue(sherlock.use_ois)
        finally:
            os.remove("tois.csv")
            os.remove("kic_star.csv")
            os.remove("kois.csv")
            os.remove("epic_ois.csv")

    def test_run_empty(self):
        sherlock = Sherlock([])
        self.assertFalse(sherlock.use_ois)
        sherlock.run()
        object_dir = "FFI_TIC 181084752_all"
        self.assertFalse(os.path.exists(object_dir))

    def test_run(self):
        sherlock = Sherlock([MissionFfiIdObjectInfo("TIC 181804752", 'all')])
        sherlock.setup_detrend(n_detrends=1, initial_smooth=False, initial_rms_mask=False)\
            .setup_transit_adjust_params(max_runs=1).run()
        self.__assert_run_files("FFI_TIC 181804752_all", assert_rms_mask=False)

    def test_run_with_rms_mask(self):
        sherlock = Sherlock([MissionFfiIdObjectInfo("TIC 181804752", 'all')])
        sherlock.setup_detrend(n_detrends=2, initial_rms_mask=True)\
            .setup_transit_adjust_params(max_runs=1).run()
        self.__assert_run_files("FFI_TIC 181804752_all")

    def __assert_run_files(self, object_dir, assert_rms_mask=True):
        run_dir = object_dir + "/1"
        periodogram_file = object_dir + "/Periodogram_FFI_TIC 181804752_all.png"
        rms_mask_file = object_dir + "/High_RMS_Mask_FFI_TIC 181804752_all.png"
        lc_file = object_dir + "/lc.csv"
        report_file = object_dir + "/FFI_TIC 181804752_all_report.log"
        candidates_csv_file = object_dir + "/candidates.csv"
        try:
            self.assertTrue(os.path.exists(run_dir))
            self.assertTrue(os.path.isfile(periodogram_file))
            if assert_rms_mask:
                self.assertTrue(os.path.isfile(rms_mask_file))
            self.assertTrue(os.path.isfile(lc_file))
            self.assertTrue(os.path.isfile(report_file))
            self.assertTrue(os.path.isfile(candidates_csv_file))
        finally:
            shutil.rmtree(object_dir, ignore_errors=True)

if __name__ == '__main__':
    unittest.main()