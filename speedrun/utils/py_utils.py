from argparse import Namespace as _argparse__Namespace
from collections import Mapping, MutableMapping
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
    COMMAND_APPEND = 'append'

    @classmethod
    def parse_command(cls, cmd):
        return cmd.split(';')

    @classmethod
    def remove_wake_keys(cls, d):
        if isinstance(d, Mapping):
            for key, value in d.items():
                if key == cls.WAKE_KEY:
                    del d[key]
                elif isinstance(value, (Mapping, list)):
                    cls.remove_wake_keys(value)
        elif isinstance(d, list):
            things_to_remove = []
            for elem in d:
                if isinstance(elem, Mapping):
                    if cls.WAKE_KEY in elem and len(elem) == 1:
                        # The dict exists for the sole purpose of telling
                        # speedrun to do something, so we clean it out
                        things_to_remove.append(elem)
                    else:
                        # The dict contains other things, so we let the
                        # recursion deal with it.
                        cls.remove_wake_keys(elem)
            for thing in things_to_remove:
                d.remove(thing)
        return d

    @classmethod
    def update_dict(cls, config, macro, copy=True):
        for macro_k, macro_v in macro.items():
            config_v = config.get(macro_k)
            # Check if we're purging/appending to existing content
            if isinstance(macro_v, Mapping) and cls.WAKE_KEY in macro_v:
                macro_command = macro_v.pop(cls.WAKE_KEY)
                purge_now = (cls.COMMAND_PURGE in cls.parse_command(macro_command) or
                             f"__{cls.COMMAND_PURGE}__" in cls.parse_command(macro_command))
            else:
                purge_now = False
            if isinstance(macro_v, list):
                append_now = {cls.WAKE_KEY: cls.COMMAND_APPEND} in macro_v
                if append_now:
                    macro_v.remove({cls.WAKE_KEY: cls.COMMAND_APPEND})
            else:
                append_now = False
            if isinstance(macro_v, Mapping) and isinstance(config_v, Mapping) and not purge_now:
                cls.update_dict(config_v, macro_v, copy=copy)
            elif isinstance(macro_v, list) and isinstance(config_v, list) and append_now:
                config[macro_k] = list(config_v) + list(macro_v)
            else:
                if copy:
                    config[macro_k] = cls.remove_wake_keys(deepcopy(macro_v))
                else:
                    config[macro_k] = cls.remove_wake_keys(macro_v)


def flatten_dict(d, parent_key='', sep='_'):
    # https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
