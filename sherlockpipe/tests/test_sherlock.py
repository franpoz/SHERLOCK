import os
import shutil
import unittest
from lcbuilder.star.starinfo import StarInfo
from lcbuilder.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from sherlockpipe.scoring.BasicSignalSelector import BasicSignalSelector
from sherlockpipe.scoring.QuorumSnrBorderCorrectedSignalSelector import QuorumSnrBorderCorrectedSignalSelector
from sherlockpipe.scoring.SnrBorderCorrectedSignalSelector import SnrBorderCorrectedSignalSelector
from sherlockpipe.sherlock import Sherlock
from sherlockpipe.sherlock_target import SherlockTarget


class TestsSherlock(unittest.TestCase):
    def test_setup_files(self):
        sherlock = Sherlock(None)
        sherlock.setup_files(False, False, False, "inner/")
        self.assertEqual("inner/", sherlock.results_dir)

    def test_setup_detrend(self):
        sherlock = Sherlock([SherlockTarget(object_info=None, smooth_enabled=False, high_rms_enabled=False,
                                                  high_rms_threshold=3, high_rms_bin_hours=9, detrends_number=2,
                                                  detrend_method="gp", cpu_cores=3, auto_detrend_enabled=True,
                                                  auto_detrend_ratio=1 / 2, auto_detrend_method="cosine")])
        self.assertEqual(False, sherlock.sherlock_targets[0].smooth_enabled)
        self.assertEqual(False, sherlock.sherlock_targets[0].high_rms_enabled)
        self.assertEqual(3, sherlock.sherlock_targets[0].high_rms_threshold)
        self.assertEqual(9, sherlock.sherlock_targets[0].high_rms_bin_hours)
        self.assertEqual(2, sherlock.sherlock_targets[0].detrends_number)
        self.assertEqual("gp", sherlock.sherlock_targets[0].detrend_method)
        self.assertEqual(3, sherlock.sherlock_targets[0].cpu_cores)
        self.assertEqual(True, sherlock.sherlock_targets[0].auto_detrend_enabled)
        self.assertEqual(1 / 2, sherlock.sherlock_targets[0].auto_detrend_ratio)
        self.assertEqual("cosine", sherlock.sherlock_targets[0].auto_detrend_method)

    def test_setup_transit_adjust_params(self):
        sherlock = Sherlock([SherlockTarget(object_info=None, max_runs=5, period_protect=5, period_min=1,
                                                   period_max=2, bin_minutes=5, cpu_cores=3, snr_min=6, sde_min=5,
                                                   mask_mode="subtract", best_signal_algorithm="quorum",
                                                   quorum_strength=2)])
        self.assertEqual(5, sherlock.sherlock_targets[0].max_runs)
        self.assertEqual(5, sherlock.sherlock_targets[0].period_protect)
        self.assertEqual(1, sherlock.sherlock_targets[0].period_min)
        self.assertEqual(2, sherlock.sherlock_targets[0].period_max)
        self.assertEqual(5, sherlock.sherlock_targets[0].bin_minutes)
        self.assertEqual(3, sherlock.sherlock_targets[0].cpu_cores)
        self.assertEqual(6, sherlock.sherlock_targets[0].snr_min)
        self.assertEqual(5, sherlock.sherlock_targets[0].sde_min)
        self.assertEqual("subtract", sherlock.sherlock_targets[0].mask_mode)
        self.assertEqual("quorum", sherlock.sherlock_targets[0].best_signal_algorithm)
        self.assertEqual(2, sherlock.sherlock_targets[0].quorum_strength)

    def test_scoring_algorithm(self):
        sherlock = Sherlock([SherlockTarget(object_info=None, best_signal_algorithm="basic")])
        self.assertTrue(isinstance(sherlock.sherlock_targets[0].signal_score_selectors["basic"], BasicSignalSelector))
        sherlock = Sherlock([SherlockTarget(object_info=None, best_signal_algorithm="border-correct")])
        self.assertTrue(isinstance(sherlock.sherlock_targets[0].signal_score_selectors["border-correct"],
                                   SnrBorderCorrectedSignalSelector))
        sherlock = Sherlock([SherlockTarget(object_info=None, best_signal_algorithm="quorum")])
        self.assertTrue(isinstance(sherlock.sherlock_targets[0].signal_score_selectors["quorum"],
                                   QuorumSnrBorderCorrectedSignalSelector))

    def test_apdate_tois(self):
        sherlock = Sherlock(None)
        sherlock.ois_manager.update_tic_csvs()
        try:
            self.assertTrue(os.path.isfile(sherlock.ois_manager.tois_csv))
        finally:
            os.remove(sherlock.ois_manager.tois_csv)

    def test_apdate_kois(self):
        sherlock = Sherlock(None)
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
        sherlock = Sherlock(None)
        sherlock.ois_manager.update_epic_csvs()
        try:
            self.assertTrue(os.path.isfile(sherlock.ois_manager.epic_csv))
        finally:
            os.remove(sherlock.ois_manager.epic_csv)

    def test_ois_loaded(self):
        sherlock = Sherlock(None)
        sherlock.load_ois(True, False, False)
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
        sherlock = Sherlock([])
        self.assertFalse(sherlock.use_ois)
        sherlock.run()
        object_dir = "TIC181084752_FFI_all"
        self.assertFalse(os.path.exists(object_dir))

    def test_run(self):
        run_dir = "TIC181804752_FFI_all"
        try:
            Sherlock([SherlockTarget(MissionFfiIdObjectInfo("TIC 181804752", 'all'),
                                                       detrends_number=1, smooth_enabled=False,
                                                       high_rms_enabled=False, max_runs=1)]).run()
            self.__assert_run_files(run_dir, assert_rms_mask=False)
        finally:
            self.__clean(run_dir)

    def test_run_with_rms_mask(self):
        run_dir = "TIC181804752_FFI_all"
        try:
            Sherlock([SherlockTarget(MissionFfiIdObjectInfo("TIC 181804752", 'all'), high_rms_enabled=True,
                                            max_runs=1)]).run()
            self.__assert_run_files(run_dir)
        finally:
            self.__clean(run_dir)

    def test_run_with_explore(self):
        run_dir = None
        try:
            Sherlock([SherlockTarget(MissionFfiIdObjectInfo("TIC 181804752", 'all'),
                                                       detrends_number=1, high_rms_enabled=True)], True).run()
            run_dir = "TIC181804752_FFI_all"
            self.assertTrue(os.path.exists(run_dir))
            self.assertTrue(os.path.exists(run_dir + "/Periodogram_TIC181804752_FFI_all.png"))
            self.assertFalse(os.path.exists(run_dir + "/1"))
        finally:
            self.__clean(run_dir)

    def test_run_with_autodetrend(self):
        run_dir = None
        try:
            Sherlock([SherlockTarget(MissionObjectInfo("TIC 259377017", [5]), detrends_number=1,
                                                       auto_detrend_enabled=True, max_runs=1)],
                     True).run()
            run_dir = "TIC259377017_[5]"
            self.assertTrue(os.path.exists(run_dir))
            self.assertTrue(os.path.exists(run_dir + "/Phase_detrend_period_TIC259377017_[5]_0.52_days.png"))
        finally:
            self.__clean(run_dir)

    def test_run_epic_ffi(self):
        run_dir = None
        try:
            Sherlock([SherlockTarget(MissionFfiIdObjectInfo("EPIC 249631677", 'all'),
                                                       detrends_number=1, high_rms_enabled=True,
                                                       auto_detrend_enabled=False, max_runs=1)], False).run()
            run_dir = "EPIC249631677_FFI_all"
            self.assertTrue(os.path.exists(run_dir))
            self.assertTrue(os.path.exists(run_dir + "/Periodogram_EPIC249631677_FFI_all.png"))
            self.assertTrue(os.path.exists(run_dir + "/1"))
        finally:
            self.__clean(run_dir)

    def test_run_with_star_info(self):
        run_dir = None
        try:
            Sherlock([SherlockTarget(MissionFfiIdObjectInfo("TIC 181804752", 'all',
                                                               star_info=StarInfo(ld_coefficients=(0.15,0.25),
                                                                                  teff=4000,
                                                                                  lum=1.50, logg=0.15, radius=0.4,
                                                                                  radius_min=0.10, radius_max=0.15,
                                                                                  mass=0.3, mass_min=0.05, mass_max=0.075,
                                                                                  ra=15, dec=87)), detrends_number=1,
                                                       high_rms_enabled=True, max_runs=1)], True).run()
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
            sherlock = Sherlock([SherlockTarget(MissionFfiIdObjectInfo("TIC 181804752", 'all'),
                                                       detrends_number=1, high_rms_enabled=True, max_runs=1,
                                                       oversampling=5.5, t0_fit_margin=0.09,
                                                       duration_grid_step=1.075, fit_method="bls",
                                                       best_signal_algorithm="quorum", quorum_strength=0.31)], False)\
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
