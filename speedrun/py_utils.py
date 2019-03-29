from argparse import Namespace as _argparse__Namespace
from pydoc import locate as loc
from importlib import util as imputils
from types import ModuleType
from importlib import import_module
import os


class Namespace(_argparse__Namespace):
    """A fancier Namespace."""

    def get(self, tag, default=None, ensure_exists=False):
        paths = tag.split("/")
        data = self.__dict__
        # noinspection PyShadowingNames
        for path in paths:
            if ensure_exists:
                assert path in data
            data = data.get(path, default if path == paths[-1] else {})
        return data

    def __getitem__(self, item):
        return self.__dict__.__getitem__(item)

    def __setitem__(self, key, value):
        self.__dict__.__setitem__(key, value)

    def update(self, with_dict):
        self.__dict__.update(with_dict)
        return self

    def set(self, tag, value):
        paths = tag.split('/')
        data = self
        for path in paths[:-1]:
            if path in data:
                data = data[path]
            else:
                data.update({path: Namespace()})
                data = data[path]
        data[paths[-1]] = value
        return self

    def new(self, tag, **kwargs):
        self.set(tag, Namespace(**kwargs))
        return self


def recursive_update_inplace(d1, d2):
    '''
    Update d1 with the data from d2 recursively
    :param d1: dict
    :param d2: dict
    :return: None
    '''
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            recursive_update_inplace(d1[key], value)
        else:
            d1[key] = value


def locate(path, import_from=None, forceload=0, ensure_exist=True):
    """pydoc locate relative to path(s) given in import_from"""
    if isinstance(import_from, (list, tuple)):
        for base in tuple(import_from) + (None,):
            obj = locate(path, base, forceload, ensure_exist=False)
            if obj is not None:
                return obj
        obj = None
    else:
        if isinstance(import_from, ModuleType):
            import_from = import_from.__name__
        assert import_from is None or isinstance(import_from, str), f'{type(import_from)}'
        if import_from is not None:
            obj = loc(import_from + '.' + path, forceload=forceload)
        else:
            obj = loc(path, forceload=forceload)
    if ensure_exist and obj is None:
        import_from = [m.__name__ if isinstance(m, ModuleType) else m for m in import_from] \
            if isinstance(import_from, (list, tuple)) else import_from
        assert False, f"Could not locate '{path}'" + (f' in {import_from}.' if import_from is not None else '.')
    return obj


def get_single_key_value_pair(d):
    """
    Returns the key and value of a one element dictionary, checking that it actually has only one element
    Parameters
    ----------
    d : dict

    Returns
    -------
    tuple

    """
    assert isinstance(d, dict), f'{d} is not a dictionary'
    assert len(d) == 1, f'{d} is not of length 1'
    return next(iter(d.items()))


def create_instance(class_dict, import_from=None):

    mclass, kwargs = get_single_key_value_pair(class_dict)

    # add extra import locations that might be specified in the class dict
    locations_from_class_dict = kwargs.pop('import_from', None)
    import_from = import_from if locations_from_class_dict is None else [locations_from_class_dict, import_from]

    # get positional arguments if specified
    args = kwargs.pop('args', [])

    network_class = locate(mclass, import_from)
    if "noargs" in kwargs:
        return network_class()
    else:
        return network_class(*args, **kwargs)


if __name__ == '__main__':
    print(locate('torch.sigmoid'))
    print(locate('sigmoid', 'torch'))
    print(locate('torch.sigmoid', 'torch'))
    print(locate('sigmoid', ['numpy', 'torch']))
    import torch as pytorch
    import numpy as np
    print(locate('sigmoid', [np, pytorch]))
    print(locate('no_such_class_or_module', [np, pytorch], ensure_exist=False))
    print(locate('no_such_class_or_module', [np, pytorch]))
