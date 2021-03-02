import os
import shutil
import unittest
from sherlockpipe.star.starinfo import StarInfo
from sherlockpipe.objectinfo.InputObjectInfo import InputObjectInfo
from sherlockpipe.objectinfo.MissionFfiCoordsObjectInfo import MissionFfiCoordsObjectInfo
from sherlockpipe.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from sherlockpipe.objectinfo.MissionInputObjectInfo import MissionInputObjectInfo
from sherlockpipe.objectinfo.MissionObjectInfo import MissionObjectInfo
from sherlockpipe.objectinfo.preparer.MissionFfiLightcurveBuilder import MissionFfiLightcurveBuilder
from sherlockpipe.objectinfo.preparer.MissionInputLightcurveBuilder import MissionInputLightcurveBuilder
from sherlockpipe.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder
from sherlockpipe.scoring.BasicSignalSelector import BasicSignalSelector
from sherlockpipe.scoring.QuorumSnrBorderCorrectedSignalSelector import QuorumSnrBorderCorrectedSignalSelector
from sherlockpipe.scoring.SnrBorderCorrectedSignalSelector import SnrBorderCorrectedSignalSelector
from sherlockpipe.sherlock import Sherlock


class TestsSherlock(unittest.TestCase):
    def test_setup_files(self):
        sherlock = Sherlock(False, None)
        sherlock.setup_files(False, "inner/")
        self.assertEqual("inner/", sherlock.results_dir)

    def test_setup_detrend(self):
        sherlock = Sherlock(False, None)
        sherlock.setup_detrend(initial_smooth=False, initial_rms_mask=False, initial_rms_threshold=3,
                               initial_rms_bin_hours=9, n_detrends=2, detrend_method="gp", cores=3,
                               auto_detrend_periodic_signals=True, auto_detrend_ratio=1 / 2,
                               auto_detrend_method="cosine")
        self.assertEqual(False, sherlock.initial_smooth)
        self.assertEqual(False, sherlock.initial_rms_mask)
        self.assertEqual(3, sherlock.initial_rms_threshold)
        self.assertEqual(9, sherlock.initial_rms_bin_hours)
        self.assertEqual(2, sherlock.n_detrends)
        self.assertEqual("gp", sherlock.detrend_method)
        self.assertEqual(3, sherlock.detrend_cores)
        self.assertEqual(True, sherlock.auto_detrend_periodic_signals)
        self.assertEqual(1 / 2, sherlock.auto_detrend_ratio)
        self.assertEqual("cosine", sherlock.auto_detrend_method)

    def test_setup_transit_adjust_params(self):
        sherlock = Sherlock(False, None)
        sherlock.setup_transit_adjust_params(max_runs=5, period_protec=5, period_min=1, period_max=2, bin_minutes=5,
                                             run_cores=3, snr_min=6, sde_min=5, fap_max=0.05, mask_mode="subtract",
                                             best_signal_algorithm="quorum", quorum_strength=2)
        self.assertEqual(5, sherlock.max_runs)
        self.assertEqual(5, sherlock.period_protec)
        self.assertEqual(1, sherlock.period_min)
        self.assertEqual(2, sherlock.period_max)
        self.assertEqual(5, sherlock.bin_minutes)
        self.assertEqual(3, sherlock.run_cores)
        self.assertEqual(6, sherlock.snr_min)
        self.assertEqual(5, sherlock.sde_min)
        self.assertEqual(0.05, sherlock.fap_max)
        self.assertEqual("subtract", sherlock.mask_mode)
        self.assertEqual("quorum", sherlock.best_signal_algorithm)
        # TODO test quorum strength

    def test_scoring_algorithm(self):
        sherlock = Sherlock(False, None)
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
        sherlock = Sherlock(False, object_info)
        self.assertTrue(isinstance(sherlock.lightcurve_builders[type(object_info)], MissionLightcurveBuilder))
        object_info = MissionFfiIdObjectInfo("TIC 1234567", 'all')
        sherlock = Sherlock(False, object_info)
        self.assertTrue(isinstance(sherlock.lightcurve_builders[type(object_info)], MissionFfiLightcurveBuilder))
        object_info = MissionFfiCoordsObjectInfo(19, 15, 'all')
        sherlock = Sherlock(False, object_info)
        self.assertTrue(isinstance(sherlock.lightcurve_builders[type(object_info)], MissionFfiLightcurveBuilder))
        object_info = MissionInputObjectInfo("TIC 1234567", "testfilename.csv")
        sherlock = Sherlock(False, object_info)
        self.assertTrue(isinstance(sherlock.lightcurve_builders[type(object_info)], MissionInputLightcurveBuilder))
        object_info = InputObjectInfo("testfilename.csv")
        sherlock = Sherlock(False, object_info)
        self.assertTrue(isinstance(sherlock.lightcurve_builders[type(object_info)], MissionInputLightcurveBuilder))

    def test_apdate_tois(self):
        sherlock = Sherlock(False, None)
        sherlock.ois_manager.update_tic_csvs()
        try:
            self.assertTrue(os.path.isfile(sherlock.ois_manager.tois_csv))
        finally:
            os.remove(sherlock.ois_manager.tois_csv)

    def test_apdate_kois(self):
        sherlock = Sherlock(False, None)
        sherlock.ois_manager.update_kic_csvs()
        try:
            self.assertTrue(os.path.isfile(sherlock.ois_manager.kic_star_csv))
        finally:
            os.remove(sherlock.ois_manager.kic_star_csv)
        try:
            self.assertTrue(os.path.isfile(sherlock.ois_manager.kois_csv))
        finally:
            os.remove(sherlock.ois_manager.kois_csv)

    def test_apdate_epicois(self):
        sherlock = Sherlock(False, None)
        sherlock.ois_manager.update_epic_csvs()
        try:
            self.assertTrue(os.path.isfile(sherlock.ois_manager.epic_csv))
        finally:
            os.remove(sherlock.ois_manager.epic_csv)

    def test_ois_loaded(self):
        sherlock = Sherlock(False, None)
        sherlock.load_ois(True)
        sherlock.filter_hj_ois()
        try:
            self.assertGreater(len(sherlock.ois.index), 100)
            sherlock.limit_ois(0, 5)
            self.assertEqual(len(sherlock.ois.index), 5)
            self.assertTrue(sherlock.use_ois)
        finally:
            os.remove(sherlock.ois_manager.tois_csv)
            os.remove(sherlock.ois_manager.ctois_csv)
            os.remove(sherlock.ois_manager.kic_star_csv)
            os.remove(sherlock.ois_manager.kois_csv)
            os.remove(sherlock.ois_manager.epic_csv)

    def test_run_empty(self):
        sherlock = Sherlock(False, [])
        self.assertFalse(sherlock.use_ois)
        sherlock.run()
        object_dir = "TIC181084752_FFI_all"
        self.assertFalse(os.path.exists(object_dir))

    def test_run(self):
        run_dir = "TIC181804752_FFI_all"
        try:
            sherlock = Sherlock(False, [MissionFfiIdObjectInfo("TIC 181804752", 'all')])
            sherlock.setup_detrend(n_detrends=1, initial_smooth=False, initial_rms_mask=False)\
                .setup_transit_adjust_params(max_runs=1).run()
            self.__assert_run_files(run_dir, assert_rms_mask=False)
        finally:
            self.__clean(run_dir)

    def test_run_with_rms_mask(self):
        run_dir = "TIC181804752_FFI_all"
        try:
            sherlock = Sherlock(False, [MissionFfiIdObjectInfo("TIC 181804752", 'all')])
            sherlock.setup_detrend(n_detrends=1, initial_rms_mask=True) \
                .setup_transit_adjust_params(max_runs=1).run()
            self.__assert_run_files(run_dir)
        finally:
            self.__clean(run_dir)

    def test_run_with_explore(self):
        run_dir = None
        try:
            sherlock = Sherlock(False, [MissionFfiIdObjectInfo("TIC 181804752", 'all')], True)
            sherlock.setup_detrend(n_detrends=1, initial_rms_mask=True) \
                .setup_transit_adjust_params(max_runs=1).run()
            run_dir = "TIC181804752_FFI_all"
            self.assertTrue(os.path.exists(run_dir))
            self.assertTrue(os.path.exists(run_dir + "/Periodogram_TIC181804752_FFI_all.png"))
            self.assertFalse(os.path.exists(run_dir + "/1"))
        finally:
            self.__clean(run_dir)

    def test_run_with_autodetrend(self):
        run_dir = None
        try:
            sherlock = Sherlock(False, [MissionObjectInfo("TIC 259377017", [5])], True)
            sherlock.setup_detrend(n_detrends=1, initial_rms_mask=True, auto_detrend_periodic_signals=True) \
                .setup_transit_adjust_params(max_runs=1).run()
            run_dir = "TIC259377017_[5]"
            self.assertTrue(os.path.exists(run_dir))
            self.assertTrue(os.path.exists(run_dir + "/Phase_detrend_period_TIC259377017_[5]_3.60_days.png"))
        finally:
            self.__clean(run_dir)

    def test_run_epic_ffi(self):
        run_dir = None
        try:
            sherlock = Sherlock(False, [MissionFfiIdObjectInfo("EPIC 249631677", 'all')], False)
            sherlock.setup_detrend(n_detrends=1, initial_rms_mask=True, auto_detrend_periodic_signals=False) \
                .setup_transit_adjust_params(max_runs=1).run()
            run_dir = "EPIC249631677_FFI_all"
            self.assertTrue(os.path.exists(run_dir))
            self.assertTrue(os.path.exists(run_dir + "/Periodogram_EPIC249631677_FFI_all.png"))
            self.assertTrue(os.path.exists(run_dir + "/1"))
        finally:
            self.__clean(run_dir)

    def test_run_with_star_info(self):
        run_dir = None
        try:
            sherlock = Sherlock(False, [MissionFfiIdObjectInfo("TIC 181804752", 'all',
                                                               star_info=StarInfo(ld_coefficients=(0.15,0.25),
                                                                                  teff=4000,
                                                                                  lum=1.50, logg=0.15, radius=0.4,
                                                                                  radius_min=0.10, radius_max=0.15,
                                                                                  mass=0.3, mass_min=0.05, mass_max=0.075,
                                                                                  ra=15, dec=87))], True)
            sherlock.setup_detrend(n_detrends=1, initial_rms_mask=True)\
                .setup_transit_adjust_params(max_runs=1).run()
            run_dir = "TIC181804752_FFI_all"
            self.assertTrue(os.path.exists(run_dir))
            self.assertTrue(os.path.exists(run_dir + "/Periodogram_TIC181804752_FFI_all.png"))
            self.assertFalse(os.path.exists(run_dir + "/1"))
            with open(run_dir + '/TIC181804752_FFI_all_report.log') as f:
                content = f.read()
                self.assertTrue('mass = 0.3' in content)
                self.assertTrue('mass_min = 0.25' in content)
                self.assertTrue('mass_max = 0.375' in content)
                self.assertTrue('radius = 0.4' in content)
                self.assertTrue('radius_min = 0.3' in content)
                self.assertTrue('radius_max = 0.55' in content)
                self.assertTrue('limb-darkening estimates using quadratic LD (a,b)= (0.15, 0.25)' in content)
                self.assertTrue('teff = 4000' in content)
                self.assertTrue('logg = 0.15' in content)
                self.assertTrue('lum = 1.50' in content)
        finally:
            self.__clean(run_dir)

    def test_run_with_transit_customs(self):
        run_dir = None
        try:
            sherlock = Sherlock(False, [MissionFfiIdObjectInfo("TIC 181804752", 'all')], False)
            sherlock.setup_detrend(n_detrends=1, initial_rms_mask=True)\
                .setup_transit_adjust_params(max_runs=1, oversampling=5.5, t0_fit_margin=0.09, duration_grid_step=1.075,
                                             fit_method="bls", best_signal_algorithm="quorum", quorum_strength=0.31)\
                .run()
            run_dir = "TIC181804752_FFI_all"
            with open(run_dir + '/TIC181804752_FFI_all_report.log') as f:
                content = f.read()
                self.assertTrue('Fit method: box' in content)
                self.assertTrue('Duration step: 1.075' in content)
                self.assertTrue('T0 Fit Margin: 0.09' in content)
                self.assertTrue('Oversampling: 5' in content)
                self.assertTrue('Signal scoring algorithm: quorum' in content)
                self.assertTrue('Quorum algorithm vote strength: 0.31' in content)
            self.__assert_run_files(run_dir)
        finally:
            self.__clean(run_dir)

    def __assert_run_files(self, object_dir, assert_rms_mask=True):
        run_dir = object_dir + "/1"
        periodogram_file = object_dir + "/Periodogram_TIC181804752_FFI_all.png"
        rms_mask_file = object_dir + "/High_RMS_Mask_TIC181804752_FFI_all.png"
        lc_file = object_dir + "/lc.csv"
        report_file = object_dir + "/TIC181804752_FFI_all_report.log"
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

    def __clean(self, run_dir):
        if run_dir is not None and os.path.isdir(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)
        mast_download_dir = "mastDownload"
        if mast_download_dir is not None and os.path.isdir(mast_download_dir):
            shutil.rmtree(mast_download_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
