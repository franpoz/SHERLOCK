import os
import shutil
import types
import unittest
import pkg_resources

from sherlockpipe.system_stability.run import run_stability
from sherlockpipe.validation.run import run_validate
from sherlockpipe.vet import run_vet


class TestsEntrypoints(unittest.TestCase):
    def test_validation(self):
        object_dir = TestsEntrypoints.get_path('test_endpoints_data/')
        args = types.SimpleNamespace()
        args.object_dir = object_dir
        args.candidate = 1
        args.cpus = 1
        args.contrast_curve = None
        args.bins = 50
        args.scenarios = 2
        args.sigma_mode = 'flux_err'
        validation_dir = object_dir + '/validation_0'
        try:
            run_validate(args)
            self.assertEquals(6, len(os.listdir(validation_dir + '/triceratops')))
        finally:
            shutil.rmtree(validation_dir, ignore_errors=True)

    def test_vetting(self):
        object_dir = TestsEntrypoints.get_path('test_endpoints_data/')
        vetting_dir = object_dir + '/vetting_0'
        try:
            run_vet(object_dir, 1, None, cpus=4)
            self.assertEquals(6, len(os.listdir(vetting_dir)))
        finally:
            shutil.rmtree(vetting_dir, ignore_errors=True)

    def test_stability(self):
        object_dir = TestsEntrypoints.get_path('test_endpoints_data/')
        properties_dir = TestsEntrypoints.get_path("test_endpoints_data/stability.yaml")
        args = types.SimpleNamespace()
        args.object_dir = object_dir
        args.properties = properties_dir
        args.cpus = 4
        args.star_mass_bins = 1
        args.period_bins = 1
        args.free_params = None
        args.use_spock = False
        args.years = 500
        try:
            run_stability(args)
            self.assertEquals(2, len(os.listdir(object_dir + '/stability_0')))
        finally:
            shutil.rmtree(object_dir + '/stability_0', ignore_errors=True)


    @staticmethod
    def get_path(path):
        """
        Gets right path for tests environment
        :param path:
        :return: the real path of the test resource
        """
        return pkg_resources.resource_filename(__name__, path)


if __name__ == '__main__':
    unittest.main()
