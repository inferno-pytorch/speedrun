import os
import shutil
import sys
import ast
import subprocess

import yaml
# This registers the constructors
from .utils.py_utils import Namespace, MacroReader

try:
    from torch import save
    from torch import load
except ImportError:
    try:
        import dill
        serializer = dill
    except ImportError:
        dill = None
        import pickle
        serializer = pickle

    def save(obj, file_path):
        with open(file_path) as f:
            serializer.dump(obj, f, protocol=serializer.HIGHEST_PROTOCOL)

    def load(file_path):
        with open(file_path) as f:
            out = serializer.load(f)
        return out


class BaseExperiment(object):
    """This could be the name of the method BaseExperiment.run() should call by default."""
    DEFAULT_DISPATCH = None

    def __init__(self, experiment_directory=None):
        """
        Base class for all experiments to derive from.

        Parameters
        ----------
        experiment_directory : str
            Specify a directory for the experiment. If it doesn't exist already, it'll be
            created along with 4 subfolders: 'Configurations', 'Logs , 'Weights' and 'Plots'.
        """
        # Privates
        self._experiment_directory = None
        self._step = None
        self._epoch = None
        self._config = {}
        self._meta_config = {'exclude_attrs_from_save': [],
                             'stateless_attributes': [],
                             'stateful_attributes': []}
        self._cache = {}
        self._argv = None
        self._default_dispatch = None
        # Publics
        self.experiment_directory = experiment_directory
        # Initialize mixin classes
        super(BaseExperiment, self).__init__()

    @property
    def step(self):
        """The current (global) step."""
        if self._step is None:
            self._step = 0
        return self._step

    def next_step(self):
        """Increments the global step counter."""
        self._step = 0 if self._step is None else self._step
        self._step += 1
        return self

    @property
    def epoch(self):
        """The current epoch."""
        if self._epoch is None:
            self._epoch = 0
        return self._epoch

    def next_epoch(self):
        """Increments the epoch counter."""
        self._epoch = 0 if self._epoch is None else self._epoch
        self._epoch += 1
        return self

    @property
    def experiment_directory(self):
        """Directory for the experiment."""
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
        """Directory where the log files go."""
        if self._experiment_directory is not None:
            return os.path.join(self._experiment_directory, 'Logs')
        else:
            return None

    @property
    def checkpoint_directory(self):
        """Directory where the checkpoints go."""
        if self._experiment_directory is not None:
            return os.path.join(self._experiment_directory, 'Weights')
        else:
            return None

    @property
    def checkpoint_path(self):
        return os.path.join(self.checkpoint_directory, f'ckpt_iter_{self.step}.pt')

    @property
    def plot_directory(self):
        """Directory where the plots go."""
        if self._experiment_directory is not None:
            return os.path.join(self._experiment_directory, 'Plots')
        else:
            return None

    @property
    def configuration_directory(self):
        """Directory where the configurations go."""
        if self._experiment_directory is not None:
            return os.path.join(self._experiment_directory, 'Configurations')
        else:
            return None

    def inherit_configuration(self, from_experiment_directory, file_name='train_config.yml',
                              read=True):
        """
        Given another experiment directory, inherit the configuration file by
        copying it over to the current configuration directory.

        Parameters
        ----------
        from_experiment_directory : str
            The other experiment directory to inherit from.
        file_name : str
            Name of the .yml configuration file.
        read : bool
            Whether to read the configuration file after copying it over.
        Returns
        -------
            BaseExperiment

        """
        source_path = os.path.join(from_experiment_directory, 'Configurations', file_name)
        target_path = os.path.join(self.configuration_directory, file_name)
        shutil.copy(source_path, target_path)
        if read:
            self.read_config_file()
        return self

    def dump_configuration(self, file_name='train_config.yml'):
        """
        Dump current configuration (dictionary) to a file in the configuration directory
        of the current experiment.

        Parameters
        ----------
        file_name : str
            Name of the .yml file to dump to.

        Returns
        -------
            BaseExperiment
        """
        dump_path = os.path.join(self.configuration_directory, file_name)
        with open(dump_path, 'w') as f:
            yaml.dump(self._config, f)
        return self

    def record_args(self):
        """Record the command line args. This must be called before calling say `get_arg`."""
        self._argv = sys.argv
        return self

    def get_arg(self, tag, default=None, ensure_exists=False):
        """
        Get command line argument.

        Parameters
        ----------
        tag : str or int
            Command line argument name or index.
        default :
            Default value.
        ensure_exists :
            Raise an error if tag not found in command line arguments.

        Examples
        --------
        In the terminal:

        $ python my_experiment.py ./EXPERIMENT-0 --blah 42

        >>> experiment = BaseExperiment().record_args()
        >>> experiment.get_arg('blah')  # Prints 42
        >>> assert isinstance(experiment.get_arg('blah'), int)  # type parsing with ast
        >>> experiment.get_arg(0)   # Prints './EXPERIMENT-0'
        """
        assert self._argv is not None, "Args not parsed yet. Have you called `self.record_args()`?"
        if not isinstance(tag, str):
            assert isinstance(tag, int)
            if ensure_exists:
                assert tag < len(self._argv), \
                    f"Accessing arg at index {tag}, but only {len(self._argv)} args available."
            return default if tag >= len(self._argv) else self._argv[tag]
        if f'--{tag}' in self._argv:
            value = self._argv[self._argv.index(f'--{tag}') + 1]
            # try to convert value to an int or a float or something
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # nope, we'll have to live with a string
                pass
            return value
        else:
            if ensure_exists:
                raise KeyError(f"Argument --{tag} is not provided, but it should be.")
            return default

    def update_configuration_from_args(self):
        """
        Override fields in the configuration file with command line arguments.

        Examples
        --------
        In the terminal:

        $ python my_experiment.py ./EXPERIMENT-0 --config.training.optimizer Adam
            --config.training.lr 0.0001

        >>> experiment = BaseExperiment().record_args().parse_experiment_directory()
        >>> experiment.read_config_file()
        >>> print(experiment.get('training/optimizer'))     # Say this prints 'SGD'
        >>> experiment.update_configuration_from_args()
        >>> print(experiment.get('training/optimizer'))     # This would print 'Adam'
        >>> print(experiment.get('training/lr'))            # This would print 0.0001
        >>> assert isinstance(experiment.get('training/lr'), float) # Works

        Returns
        -------
            BaseExperiment

        """
        for arg in self._argv:
            if arg.startswith('--config.'):
                tag = arg.replace('--config.', '').replace('.', '/')
                value = self.get_arg(arg.lstrip('--'), ensure_exists=True)
                self.set(tag, value)
        return self

    def register_unpickleable(self, *attributes):
        """
        Specify the attributes that are not pickleable. If the experiment contains
        unregistered unpickleable attributes, `BaseExperiment.checkpoint` might throw an error
        if not overloaded.
        """
        self._meta_config['exclude_attrs_from_save'].extend(list(attributes))
        return self

    def checkpoint(self, force=True):
        """
        Makes a checkpoint by dumping all experiment attributes to a pickle file.
        If force=False, this would only save if the global step and the save frequency
        set in the config file (under 'training/checkpoint_every') match.

        Warnings
        --------

        If your experiment has unpickleable objects, make sure to register them with
        `self.register_unpickleable` to not have them pickled.

        Parameters
        ----------
        force : bool
            If set to false, a checkpoint will be created only if global step and the
            save frequency set in the config file (under 'training/checkpoint_every') match.

        Returns
        -------
            BaseExperiment
        """
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

    def load_from_checkpoint(self, step=None):
        """
        Load checkpoint from file. Note that attributes registered as unpickleable are not
        pickled, and will not be loaded.

        Parameters
        ----------
        step : int
            Load checkpoint made at step.

        Returns
        -------
            BaseExperiment
        """
        for filename in os.listdir(self.checkpoint_directory):
            if filename.startswith('checkpoint_iter_') and filename.endswith('.pt'):
                try:
                    ckpt_step = int(filename.strip('checkpoint_iter_.pt'))
                except ValueError:
                    continue
                if ckpt_step == step:
                    # load
                    self_dict = load(filename)
                    self.__dict__.update(self_dict)
                    break
        else:
            raise FileNotFoundError(f"No checkpoint for step {step} found in "
                                    f"{self.checkpoint_directory}.")
        return self

    def get(self, tag, default=None, ensure_exists=False):
        """
        Retrieves a field from the configuration.

        Examples
        --------
        Say the configuration file reads:

        ```yaml
        my_field:
          my_subfield: 12
          subsubfields:
            my_subsubfield: 42
        my_new_field: 0
        ```

        >>> experiment = BaseExperiment().parse_experiment_directory().read_config_file()
        >>> print(experiment.get('my_field/my_subfield'))   # Prints 12
        >>> print(experiment.get('my_field/subsubfields/my_subsubfield'))   # Prints 42
        >>> print(experiment.get('my_new_field'))   # Prints 0
        >>> print(experiment.get('i_dont_exist', 13))   # Prints 13
        >>> print(experiment.get('i_should_exist', ensure_exists=True)) # Raises an error

        Parameters
        ----------
        tag : str
            Path in the hierarchical configuration (see example).
        default :
            Default value if object corresponding to path not found.
        ensure_exists : bool
            Whether an error should be raised if the path doesn't exist.
        """
        paths = tag.split("/")
        data = self._config
        # noinspection PyShadowingNames
        for path in paths:
            if ensure_exists:
                assert path in data
            data = data.get(path, default if path == paths[-1] else {})
        return data

    def set(self, tag, value):
        """
        Like get, but sets.

        Examples
        --------
        >>> experiment = BaseExperiment()
        >>> experiment.set('a/b', 42)
        >>> print(experiment.get('a/b'))    # Prints 42

        Parameters
        ----------
        tag : str
            Path in the hierarchical configuration.
        value :
            Value to set.

        Returns
        -------
            BaseExperiment
        """
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

    @property
    def cache_keys(self):
        """
        List keys in the cache.
        """
        return list(self._cache.keys())

    def read_from_cache(self, tag, default=None, ensure_exists=False):
        """
        Read from the cache.

        Parameters
        ----------
        tag : str
            Tag to read.
        default :
            Default value.
        ensure_exists :
            Raises an error if tag is not found in cache.

        Returns
        -------
            Cache contents.
        """
        if ensure_exists:
            assert tag in self._cache
        return self._cache.get(tag, default)

    def write_to_cache(self, tag, value):
        """
        Write a value to cache.

        Parameters
        ----------
        tag : str
            Tag to write.
        value :
            Value to write

        Returns
        -------
            BaseExperiment

        Examples
        --------
        >>> experiment = BaseExperiment()
        >>> experiment.write_to_cache('blah', 42)
        >>> experiment.read_from_cache('blah')  # Prints 42
        """
        self._cache.update({tag: value})
        return self

    def accumulate_in_cache(self, tag, value, accumulate_fn=None):
        """
        Accumulate to an object in cache.

        Parameters
        ----------
        tag : str
            Tag to accumulate to.
        value :
            Value to accumulate.
        accumulate_fn : callable
            Accumulator function. Defaults to the increment operator, +=.

        Examples
        -------
        Simple:
        >>> experiment = BaseExperiment()
        >>> experiment.write_to_cache('loss', 2)
        >>> experiment.accumulate_in_cache('loss', 1).read_from_cache('loss') # Prints 3
        >>> experiment.accumulate_in_cache('loss', 2).read_from_cache('loss') # Prints 5

        With an accumulator function:
        >>> experiment = BaseExperiment()
        >>> experiment.write_to_cache('a', {})
        >>> experiment.accumulate_in_cache('a', 3, accumulate_fn=lambda x, y: x.update({y: 'la'}))
        >>> experiment.read_from_cache('a')[3]  # Prints 'la'

        Returns
        -------
            BaseExperiment
        """
        if tag not in self._cache:
            self.write_to_cache(tag, value)
        else:
            if accumulate_fn is None:
                self._cache[tag] += value
            else:
                assert callable(accumulate_fn)
                self._cache[tag] = accumulate_fn(self._cache[tag], value)
        return self

    def clear_in_cache(self, tag):
        """Remove `tag` from cache."""
        if tag not in self._cache:
            pass
        else:
            del self._cache[tag]
        return self

    def clear_cache(self):
        """Delete everything in cache."""
        self._cache.clear()
        return self

    def bundle(self, **kwargs):
        """Pack kwargs to a Namespace object."""
        return Namespace(**kwargs)

    def read_config_file(self, file_name='train_config.yml', path=None):
        """
        Read configuration from a YAML file.

        Parameters
        ----------
        file_name : str
            Name of the file. Defaults to `train_config.yml`.
        path : str
            Path to file. Defaults to 'experiment_directory/Configurations/file_name'.

        Returns
        -------
            BaseExperiment
        """
        path = os.path.join(self.configuration_directory, file_name) if path is None else path
        if not os.path.exists(path):
            raise FileNotFoundError
        with open(path, 'r') as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)
        return self

    def read_macro(self, path=None):
        """
        If multiple experiments share the same set of command line args, it can be annoying
        to key them in manually every time. Macros is a secondary config file that can be shared
        between experiments. The main config is updated with the contents of the secondary config,
        which we call a _macro_.

        Parameters
        ----------
        path : str
            Path to the macro file. If None, it's read from the command line arg '--macro'.
            If that doesn't work, this function does nothing.

        Notes
        -----
        The `path` argument can either be a single path or a list of paths delimited by a colon.
        In other words, the following would work:
            $ python experiment.py ... --macro path/to/macro1.yml:path/to/macro2.yml

        Returns
        -------
            BaseExperiment
        """
        if path is None:
            path = self.get_arg('macro')
        if path is None:
            return
        for _path in path.split(":"):
            with open(_path, 'r') as f:
                macro = yaml.load(f, Loader=yaml.FullLoader)
            # Update config with macro
            MacroReader.update_dict(self._config, macro, copy=False)
        # Done
        return self

    def parse_experiment_directory(self):
        """Read path to experiment directory from command line arguments."""
        experiment_directory = self.get_arg(1)
        if experiment_directory is None:
            raise RuntimeError("Can't find experiment directory in command line args.")
        self.experiment_directory = experiment_directory
        return self

    def purge_existing_experiment_directory(self, experiment_directory=None):
        experiment_directory = self.get_arg(1) \
            if experiment_directory is None else experiment_directory
        if experiment_directory is None:
            raise RuntimeError("No experiment directory found to be purged.")
        if os.path.exists(experiment_directory):
            shutil.rmtree(experiment_directory)
        return self

    @staticmethod
    def register_hook(fn, key):
        setattr(fn, f'__is_speedrun_{key}_hook', True)
        return fn

    def execute_hooks(self, key):
        hook_names = [attry for attry in dir(type(self))
                      if getattr(getattr(type(self), attry), f'__is_speedrun_{key}_hook', False)]
        return {hook_name: getattr(self, hook_name)() for hook_name in hook_names}

    def run(self, *args, **kwargs):
        """
        Run the experiment. If '--dispatch method' is given as a command line argument, it's
        called with the `args` and `kwargs` provided, as in `self.method(*args, **kwargs)`.

        Say the BaseExperiment instance `my_experiment` a method called `train`,
        and it's defined in some `experiment.py` where `my_experiment.run()` is called.
        Calling `python experiment.py --dispatch train` from the command line
        will cause this method to call `my_experiment.train()`.

        In addition, this function will also run any pre-dispatch hooks if registered
        (via the `register_pre_dispatch_hook` decorator).
        """
        try:
            # Run the pre-dispatch hooks and dispatch.
            self.execute_pre_dispatch_hooks()
            return self.dispatch(self.get_dispatch_key(), *args, **kwargs)
        finally:
            self.clean_up()

    def dispatch(self, key, *args, **kwargs):
        """Dispatches a method given its name as `key`."""
        assert hasattr(self, key), f"Trying to dispatch method {key}, but it doesn't exist."
        return getattr(self, key)(*args, **kwargs)

    def get_dispatch_key(self):
        """
        Figures out what function to dispatch.
        Looks for it in the commandline args, instance attribute, decorated functions and class attribute,
        in that order.
        """
        # First look for commandline args
        if self._argv is not None and self.get_arg('dispatch', None) is not None:
            return self.get_arg('dispatch', ensure_exists=True)
        elif self.find_default_dispatch() is not None:
            return self.find_default_dispatch()
        elif self._default_dispatch is not None:
            # If that fails, check if the instance defines a default dispatch
            return self._default_dispatch
        elif self.DEFAULT_DISPATCH is not None:
            # If even that fails, use the class defined default dispatch
            return self.DEFAULT_DISPATCH
        else:
            raise RuntimeError("No default dispatch could be found. Please set it first.")

    @staticmethod
    def register_default_dispatch(fn):
        """
        Decorator to mark a method to be dispatched by default.

        Examples
        --------
        >>> @BaseExperiment.register_default_dispatch
        ... def my_default_method(self, *args):
        ...     return ...
        """
        setattr(fn, '__is_speedrun_default_dispatch', True)
        return fn

    def set_default_dispatch(self, method_name):
        """
        Set the default dispatch for _this_ instance.

        Parameters
        ----------
        method_name : str
            name of the function that will be dispatched by default.

        Returns
        -------
            BaseExperiment
        """
        assert method_name in dir(type(self)), f"Method name {method_name} not found in list of attributes."
        assert callable(getattr(type(self), method_name)), f"Default dispatch method name {method_name} should be callable."
        self._default_dispatch = method_name
        return self

    def get_default_dispatch(self):
        """Get the name of the method used as the default dispatch."""
        return self._default_dispatch

    def find_default_dispatch(self):
        """Find the name of the function marked as default dispatch."""
        for attry in dir(type(self)):
            if getattr(getattr(type(self), attry), '__is_speedrun_default_dispatch', False):
                return attry

    @staticmethod
    def register_pre_dispatch_hook(fn):
        """
        Decorator to mark a method as a pre-dispatch hook. Pre-dispatch hooks are run before the
        function being dispatched is called.
        """
        return BaseExperiment.register_hook(fn, 'pre_dispatch')

    def execute_pre_dispatch_hooks(self):
        """Execute the pre-dispatch hooks, if available. See also: `register_pre_dispatch_hook`."""
        return self.execute_hooks('pre_dispatch')

    def clean_up(self):
        """
        Overridable method to clean up the mess before exiting the process. This method is
        *guaranteed* to be called if `BaseExperiment.run` is used to dispatch (even if the dispatch
        is not successful).
        """
        pass

    def update_git_revision(self, overwrite=False):
        """
        Updates the configuration with a 'git_rev' field with the current HEAD revision.

        Parameters
        ----------
        overwrite : bool
            If a 'git_rev' field already exists, Whether to overwrite it.

        Returns
        -------
            BaseExperiment
        """
        try:
            gitcmd = ["git", "rev-parse", "--verify", "HEAD"]
            gitrev = subprocess.check_output(gitcmd).decode('latin1').strip()
        except subprocess.CalledProcessError:
            gitrev = "none"
        if not overwrite and self.get('git_rev', None) is not None:
            # Git rev already in config and we're not overwriting, so...
            pass
        else:
            self.set("git_rev", gitrev)
        return self

    def auto_setup(self, update_git_revision=True, dump_configuration=True):
        """
        Set things up automagically.

        Parameters
        ----------
        update_git_revision : bool
            Whether to update current configuration with the git revision hash.
        dump_configuration : bool
            Whether to update the configuration in file.

        Examples
        --------
        In python file experiment.py:
            >>> experiment = BaseExperiment().auto_setup()
        Let's say the experiment uses the following tags from the config file:
            >>> experiment.get('optimizer/name')
            >>> experiment.get('hyperparameters/lambda')

        As command line arguments, if you pass:
            $ python experiment.py /path/to/experiment/directory
                --inherit /path/to/previous/experiment/directory
                --config.optimizer.name RMSprop
                --config.hyperparameters.lambda 0.001
        ... the following happens.
        1. The configuration file loaded from
           `/path/to/previous/experiment/directory/Configurations/train_config.yml`
        2. The fields 'optimizer/name' and 'hyperparameters/lambda' are overwritten with the
           provided values ('RMSprop' and 0.001 respectively.)
        3. The resulting new configuration is dumped to
           `/path/to/experiment/directory/Configurations/train_config.yml`.

        Returns
        -------
            BaseExperiment
        """
        self.record_args()
        if self.get_arg('purge', False):
            self.purge_existing_experiment_directory()
        self.parse_experiment_directory()
        inherit_from = self.get_arg('inherit')
        if inherit_from is not None:
            # Inherit configuration file
            self.inherit_configuration(inherit_from, read=False)
        try:
            self.read_config_file()
        except FileNotFoundError:
            # No config file found, experiment._config remains an empty dict.
            pass
        # Read macro if available
        self.read_macro()
        # Update config from commandline args
        self.update_configuration_from_args()
        if update_git_revision:
            # Include git revision in config file
            self.update_git_revision()
        if dump_configuration:
            # Dump final config file
            self.dump_configuration()
        # Done
        return self


register_default_dispatch = BaseExperiment.register_default_dispatch
register_pre_dispatch_hook = BaseExperiment.register_pre_dispatch_hook