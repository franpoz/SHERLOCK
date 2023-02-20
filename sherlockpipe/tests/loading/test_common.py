import unittest

import pkg_resources

from sherlockpipe.loading import common


class TestCommon(unittest.TestCase):
    MODULE_PATH = pkg_resources.resource_filename("sherlockpipe.tests.resources", "RandomSignalSelector.py")
    YAML_PATH = pkg_resources.resource_filename("sherlockpipe.tests.resources", "user-properties.yaml")

    def test_load_module(self):
        loaded_module = common.load_module(self.MODULE_PATH)
        self.assertIsNotNone(loaded_module)

    def test_extract_custom_class(self):
        loaded_class = common.extract_custom_class(self.MODULE_PATH)
        self.assertIsNotNone(loaded_class)

    def test_get_from_dict(self):
        properties = common.load_from_yaml(self.YAML_PATH)
        targets = common.get_from_dict(properties, 'TARGETS')
        snr_min = common.get_from_dict(properties, 'SNR_MIN')
        self.assertIn('TIC 305048087', targets)
        self.assertIsInstance(targets, dict)
        self.assertIsInstance(snr_min, (int, float))

    def test_get_from_dict_or_default(self):
        properties = common.load_from_yaml(self.YAML_PATH)
        expected_value = 'mozart'
        value = common.get_from_dict_or_default(properties, 'AUTHOR', expected_value)
        self.assertEqual(expected_value, value)

    def test_get_from_user_or_config(self):
        properties = common.load_from_yaml(self.YAML_PATH)
        target = common.get_from_dict(properties, 'TARGETS')
        target = common.get_from_dict(target, 'TIC 305048087')
        expected_root_value = 5
        expected_value = 6
        value = common.get_from_dict(properties, 'SNR_MIN')
        self.assertEqual(expected_root_value, value)
        value = common.get_from_user_or_config(target, properties, 'SNR_MIN')
        self.assertEqual(expected_value, value)

    def test_get_from_user_or_config_or_default(self):
        properties = common.load_from_yaml(self.YAML_PATH)
        target = common.get_from_dict(properties, 'TARGETS')
        target = common.get_from_dict(target, 'TIC 305048087')
        expected_root_value = None
        expected_inner_value = None
        expected_value = 'ION'
        value = common.get_from_dict(properties, 'INVENT')
        self.assertEqual(expected_root_value, value)
        value = common.get_from_user_or_config(target, properties, 'INVENT')
        self.assertEqual(expected_inner_value, value)
        value = common.get_from_user_or_config_or_default(target, properties, 'INVENT', expected_value)
        self.assertEqual(expected_value, value)

    def test_load_from_yaml(self):
        properties = common.load_from_yaml(self.YAML_PATH)
        self.assertIsInstance(properties, dict)
        self.assertIn('TARGETS', properties)
