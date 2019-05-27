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


class MacroReader(object):
    WAKE_KEY = '__speedrun__'
    # Commands
    COMMAND_PURGE = 'purge'

    @classmethod
    def parse_command(cls, cmd):
        return cmd.split(';')

    @classmethod
    def update_dict(cls, config, macro, copy=True):
        for macro_k, macro_v in macro.items():
            config_v = config.get(macro_k)
            # Check if we're purging existing content
            if isinstance(macro_v, Mapping) and cls.WAKE_KEY in macro_v:
                macro_command = macro_v.pop(cls.WAKE_KEY)
                purge_now = 'purge' in cls.parse_command(macro_command)
            else:
                purge_now = False
            if isinstance(macro_v, Mapping) and isinstance(config_v, Mapping) and not purge_now:
                cls.update_dict(config_v, macro_v)
            else:
                if copy:
                    config[macro_k] = deepcopy(macro_v)
                else:
                    config[macro_k] = macro_v
