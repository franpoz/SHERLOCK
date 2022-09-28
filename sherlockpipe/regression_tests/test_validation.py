import os
import shutil
import unittest
import pandas as pd
import pkg_resources

from sherlockpipe.validate import Validator


class TestsValidation(unittest.TestCase):
    def test_validation(self):
        object_dir = TestsValidation.get_path('test_validation_data/')
        validation_dir = object_dir + '/validation_0'
        star_df = pd.read_csv(object_dir + "/params_star.csv")
        candidates = pd.read_csv(object_dir + "/candidates.csv")
        candidate_selection = 1
        candidates = candidates.rename(columns={'Object Id': 'id'})
        candidate = candidates.iloc[[candidate_selection - 1]]
        candidate['number'] = [candidate_selection]
        validator = Validator(object_dir, validation_dir, True, candidates)
        validator.validate(candidate, star_df.iloc[0], 1, None, 200, 1, 'flux_err')
        try:
            self.assertEquals(9, len(os.listdir(validation_dir + '/triceratops')))
        finally:
            shutil.rmtree(validation_dir + '/triceratops', ignore_errors=True)

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
