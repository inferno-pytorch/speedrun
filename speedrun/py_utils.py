from argparse import Namespace as _argparse__Namespace
from collections import Mapping
from copy import deepcopy


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


def update_nested_dict(this, other, copy=True):
    for k, v in other.items():
        d_v = this.get(k)
        if isinstance(v, Mapping) and isinstance(d_v, Mapping):
            update_nested_dict(d_v, v)
        else:
            if copy:
                this[k] = deepcopy(v)
            else:
                this[k] = v
