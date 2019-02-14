from argparse import Namespace as _argparse__Namespace
from pydoc import locate as loc
from types import ModuleType


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
