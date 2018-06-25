import os
import yaml
import argparse
import shutil
import sys
import ast

try:
    from torch import save
except ImportError:
    import dill

    def save(obj, file_path):
        with open(file_path) as f:
            dill.dump(obj, f, protocol=dill.HIGHEST_PROTOCOL)


class Add(object):
    def __init__(self, *values):
        self.values = tuple(values)

    def __int__(self):
        return int(sum(self.values))

    def __float__(self):
        return float(sum(self.values))

    @classmethod
    def from_yaml(cls, loader, node):
        values = loader.construct_sequence(node)
        return cls(*values)


yaml.add_constructor('!Add', Add.from_yaml)


class BaseExperiment(object):
    def __init__(self, experiment_directory=None):
        # Privates
        self._experiment_directory = None
        self._step = None
        self._config = {}
        self._meta_config = {'exclude_attrs_from_save': []}
        self._cache = {}
        self._argv = None
        # Publics
        self.experiment_directory = experiment_directory
        # Initialize mixin classes
        super(BaseExperiment, self).__init__()

    @property
    def step(self):
        if self._step is None:
            self._step = 0
        return self._step

    def next_step(self):
        self._step = 0 if self._step is None else self._step
        self._step += 1
        return self

    @property
    def experiment_directory(self):
        return self._experiment_directory

    @experiment_directory.setter
    def experiment_directory(self, value):
        if value is not None:
            # Make directories
            os.makedirs(os.path.join(value, 'Configurations'), exist_ok=True)
            os.makedirs(os.path.join(value, 'Logs'), exist_ok=True)
            os.makedirs(os.path.join(value, 'Weights'), exist_ok=True)
            os.makedirs(os.path.join(value, 'Plots'), exist_ok=True)
            self._experiment_directory = value

    @property
    def log_directory(self):
        if self._experiment_directory is not None:
            return os.path.join(self._experiment_directory, 'Logs')
        else:
            return None

    @property
    def checkpoint_directory(self):
        if self._experiment_directory is not None:
            return os.path.join(self._experiment_directory, 'Weights')
        else:
            return None

    @property
    def plot_directory(self):
        if self._experiment_directory is not None:
            return os.path.join(self._experiment_directory, 'Plots')
        else:
            return None

    @property
    def configuration_directory(self):
        if self._experiment_directory is not None:
            return os.path.join(self._experiment_directory, 'Configurations')
        else:
            return None

    def inherit_configuration(self, from_experiment_directory, file_name='train_config.yml'):
        source_path = os.path.join(from_experiment_directory, 'Configurations', file_name)
        target_path = os.path.join(self.configuration_directory, file_name)
        shutil.copy(source_path, target_path)
        return self

    def dump_configuration(self, file_name='train_config.yml'):
        dump_path = os.path.join(self.configuration_directory, file_name)
        with open(dump_path, 'w') as f:
            yaml.dump(self._config, f)
        return self

    def record_args(self):
        self._argv = sys.argv
        return self

    def get_arg(self, tag, default=None, ensure_exists=False):
        if f'--{tag}' in self._argv:
            value = self._argv[self._argv.index(f'--{tag}') + 1]
            # try to convert value to an int or a float or something
            try:
                value = ast.literal_eval(value)
            except ValueError:
                # nope, we'll have to live with a string
                pass
            return value
        else:
            if ensure_exists:
                raise KeyError
            return default

    def update_configuration_from_args(self):
        for arg in self._argv:
            if arg.startswith('--config.'):
                tag = arg.lstrip('--config.').replace('/')
                value = self.get_arg(arg.lstrip('--'), ensure_exists=True)
                self.set(tag, value)

    def checkpoint(self, force=True):
        if force:
            do_checkpoint = True
        else:
            do_checkpoint = (self.step % self.get('training/checkpoint_every')) == 0
        if do_checkpoint:
            self_dict = {key: val for key, val in self.__dict__.items()
                         if key not in self._meta_config['exclude_attrs_from_save']}
            save(self_dict, os.path.join(self.checkpoint_directory,
                                         f'checkpoint_iter_{self.step}.pt'))
        return self

    def get(self, tag, default=None, ensure_exists=False):
        paths = tag.split("/")
        data = self._config
        # noinspection PyShadowingNames
        for path in paths:
            if ensure_exists:
                assert path in data
            data = data.get(path, default if path == paths[-1] else {})
        return data

    def set(self, tag, value):
        paths = tag.split('/')
        data = self._config
        for path in paths[:-1]:
            if path in data:
                data = data[path]
            else:
                data.update({path: {}})
                data = data[path]
        data[paths[-1]] = value
        return self

    def read(self, tag, default=None, ensure_exists=False):
        if ensure_exists:
            assert tag in self._cache
        return self._cache.get(tag, default)

    def write(self, tag, value):
        self._cache.update({tag: value})
        return self

    def accumulate(self, tag, value, accumulate_fn=None):
        if tag not in self._cache:
            self.write(tag, value)
        else:
            if accumulate_fn is None:
                self._cache[tag] += value
            else:
                assert callable(accumulate_fn)
                self._cache[tag] = accumulate_fn(self._cache[tag], value)
        return self

    def clear(self, tag):
        if tag not in self._cache:
            pass
        else:
            del self._cache[tag]
        return self

    def clear_all(self):
        self._cache.clear()
        return self

    def read_config_file(self, file_name='train_config.yml', path=None):
        path = os.path.join(self.configuration_directory, file_name) if path is None else path
        with open(path, 'r') as f:
            self._config = yaml.load(f)
        return self

    def parse_args(self):
        # Parse args
        parsey = argparse.ArgumentParser()
        parsey.add_argument('-experiment-directory', type=str, help='Experiment directory.')
        args = parsey.parse_args()
        # Set and return
        self.experiment_directory = args.experiment_directory
        return self

    def run(self):
        raise NotImplementedError


