import pickle
import sys
import yaml
from pathlib import Path
import importlib.util


"""Includes common functions for the main entrypoints"""


def load_module(module_path: str) -> object:
    """
    Allows the dynamic load of a python module into the system

    :param str module_path: the module directory
    :return object: the module object
    """
    spec = importlib.util.spec_from_file_location("customs", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def extract_custom_class(module_path: str) -> object:
    """
    Given a module path, it loads it dynamically and extracts its inner class, assuming that the module and the class
    have the same name.

    :param str module_path: the module directory
    :return object: the class module object
    """
    class_module = None
    if module_path is not None:
        class_module = load_module(module_path)
        class_name = Path(module_path.replace(".py", "")).name
        class_module = getattr(class_module, class_name)
        globals()[class_name] = class_module
        pickle.dumps(class_module)
        class_module = class_module()
    return class_module


def get_from_dict(target: dict, key: str) -> object:
    """
    Given a target dictionary, it retrieves the value associated to a `key`

    :param dict target: the dictionary
    :param str key: the key
    :return object: the value of the key
    """
    value = None
    if isinstance(target, dict) and key in target:
        value = target[key]
    return value


def get_from_dict_or_default(target: dict, key: str, default) -> object:
    """
    This method does the same :func:`get_from_user`, but returns a default value if the key had none.

    :param dict target: the dictionary
    :param key key: the key
    :param default: the default value if nothing is found
    :return object: the final value for the key
    """
    value = get_from_dict(target, key)
    return value if value is not None else default


def get_from_user_or_config(target: dict, user_properties: dict, key: str) -> object:
    """
    Given the root user_properties dictionary and the specific target dictionary, it loads the property with name `key`
    iteratively: first tries to load the value from `user_properties`, then it tries to load it from `target`. If
    `target` contained a value, the one from `user_properties` is overwritten.

    :param dict target: the target inner properties
    :param dict user_properties: the global properties
    :param str key: the key of the properties to be extracted
    :return object: the final value
    """
    value = get_from_dict(user_properties, key)
    value = get_from_dict_or_default(target, key, value)
    return value


def get_from_user_or_config_or_default(target: dict, user_properties: dict, key: str, default) -> object:
    """
    This method does the same as :func:`get_from_user_or_config` but also returns a default value if no value was
    found associated to the given key

    :param dict target: the target inner properties
    :param dict user_properties: the global properties
    :param str key: the key of the properties to be extracted
    :param default: the default value if none was found
    :return object: the final value
    """
    value = get_from_user_or_config(target, user_properties, key)
    return value if value is not None else default


def load_from_yaml(file: str) -> dict:
    """
    Loads a yaml file into a dictionary

    :param file: the input file name
    :return dict: the dictionary
    """
    return yaml.load(open(file), yaml.SafeLoader)
