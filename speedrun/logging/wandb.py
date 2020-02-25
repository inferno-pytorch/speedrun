import os
import shutil
import yaml
from ..core import BaseExperiment
from ..utils.py_utils import flatten_dict
from ..utils.yaml_utils import read_yaml, dump_yaml
from contextlib import contextmanager
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None


class WandBMixin(object):
    WANDB_JOB_TYPE = 'train'
    WANDB_PROJECT = None

    @property
    def wandb_directory(self):
        directory = os.path.join(self.experiment_directory, 'WandB')
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return directory

    @property
    def wandb_run(self):
        return getattr(self, '_wandb_run', None)

    @wandb_run.setter
    def wandb_run(self, value):
        setattr(self, '_wandb_run', value)

    @property
    def wandb_config(self):
        return flatten_dict(self._config, sep='__')

    def initialize_wandb(self, resume=False):
        assert wandb is not None, "Install wandb first!"
        # If wandb is already initialized, ignore this call
        if self.wandb_run is not None:
            return self
        assert self.WANDB_PROJECT is not None, "Please set self.WANDB_PROJECT to use wandb."
        # If resuming, get the wandb id
        if resume:
            run_id = self.find_existing_wandb_run_id()
            assert run_id is not None, "WandB run_id could not be identified. Set WANDB_RUN_ID environment variable."
        else:
            run_id = None
        run = wandb.init(job_type=self.WANDB_JOB_TYPE, dir=self.wandb_directory, resume=resume,
                         project=self.WANDB_PROJECT, config=self.wandb_config, id=run_id)
        self.wandb_run = run
        # Dump all wandb info to file
        self.dump_wandb_info()
        return self

    @property
    def wandb_run_id(self):
        return wandb.env.get_run()

    def find_existing_wandb_run_id(self):
        run_id = None
        # Look for run_id environment variables, in the current directory, in inherited directory -- in that order.
        if wandb.env.RUN_ID in os.environ:
            run_id = os.environ[wandb.env.RUN_ID]
        elif os.path.exists(os.path.join(self.log_directory, 'wandb_info.yml')):
            # Read it in
            with open(os.path.join(self.log_directory, 'wandb_info.yml'), 'r') as f:
                info = yaml.load(f, Loader=yaml.FullLoader)
            run_id = info[wandb.env.RUN_ID]
        elif self.get_arg('inherit', None) is not None:
            potential_wandb_info_path = os.path.join(self.get_arg('inherit'), 'Logs', 'wandb_info.yml')
            if os.path.exists(potential_wandb_info_path):
                with open(potential_wandb_info_path, 'r') as f:
                    info = yaml.load(f, Loader=yaml.FullLoader)
                run_id = info[wandb.env.RUN_ID]
        return run_id

    def dump_wandb_info(self):
        info = {key: os.environ[key] for key in os.environ if key.startswith('WANDB_')}
        with open(os.path.join(self.log_directory, 'wandb_info.yml'), 'w') as f:
            yaml.dump(info, f)
        return self

    @contextmanager
    def wandb_hold_step(self):
        self.wandb_pause_step_counter()
        yield
        self.wandb_resume_step_counter()

    def wandb_pause_step_counter(self):
        setattr(self, '_wandb_step', self.wandb_run.step)
        return self

    def wandb_resume_step_counter(self):
        setattr(self, '_wandb_step', None)
        return self

    @staticmethod
    def as_wandb_image(value, image_format='chw', caption=None):
        if not isinstance(value, np.ndarray):
            # value can be a torch tensor, but we don't want to import torch.
            try:
                value = value.detach().cpu().numpy()
            except AttributeError:
                raise TypeError(f"value must be a numpy ndarray. Got {type(value).__name__} instead.")
        # Allow for value to be two dimensional
        if value.ndim == 2:
            value = value[None]
        # Convert to hwc
        if image_format.lower() == 'chw':
            value = np.moveaxis(value, 0, 2)
        elif image_format.lower() == 'hwc':
            pass
        else:
            raise ValueError(f"image_format must be one of chw or hwc; got {image_format} instead.")
        return wandb.Image(value, caption=caption)

    def wandb_log(self, **metrics):
        log = {'step': self.step, 'epoch': self.epoch}
        log.update(metrics)
        wandb.log(log, step=getattr(self, '_wandb_step', None))
        return self

    def wandb_log_scalar(self, tag, value):
        wandb.log({tag: value}, step=getattr(self, '_wandb_step', None))
        return self

    def wandb_log_image(self, tag, value, image_format='chw', caption=None):
        # Do the logging
        wandb.log({tag: self.as_wandb_image(value, image_format, caption)}, step=getattr(self, '_wandb_step', None))
        return self

    def wandb_watch(self, model, criterion=None, log='gradients', log_freq=100):
        idx = len(getattr(self, '_wandb_graphs', []))
        graph = wandb.watch(model, criterion, log, log_freq, idx=idx)
        if not hasattr(self, '_wandb_graphs'):
            setattr(self, '_wandb_graphs', [graph])
        else:
            getattr(self, '_wandb_graphs').append(graph)
        return graph

    @property
    def log_wandb_now(self):
        frequency = self.get('wandb/log_every', None)
        if frequency is not None:
            return (self.step % frequency) == 0
        else:
            return False


class WandBSweepMixin(WandBMixin):

    @property
    def wandb_sweep_id(self):
        return getattr(self, '_wandb_sweep_id', None)

    @wandb_sweep_id.setter
    def wandb_sweep_id(self, value):
        setattr(self, '_wandb_sweep_id', value)

    def setup_wandb_sweep(self):
        # Parse sweep config path and read in the config if possible.
        sweep_config_path = self.get_arg('wandb.sweep_config', ensure_exists=True)
        if not os.path.exists(sweep_config_path):
            raise FileNotFoundError(f"The file {sweep_config_path} does not exist.")
        sweep_config = read_yaml(sweep_config_path)
        # Set sweep id, dump info to file and exit.
        self.wandb_sweep_id = sweep_id = wandb.sweep(sweep_config, project=self.WANDB_PROJECT)
        dump_yaml({'wandb_sweep_id': sweep_id}, os.path.join(self.configuration_directory, 'wandb_sweep_info.yml'))
        dump_yaml(sweep_config, os.path.join(self.configuration_directory, 'wandb_sweep_config.yml'))
        return sweep_id

    def inherit_configuration(self, from_experiment_directory, file_name='train_config.yml', read=True):
        if self.get_arg('wandb.sweep', False):
            sweep_file_names = ['wandb_sweep_info.yml', 'wandb_sweep_config.yml']
            for _sweep_file_name in sweep_file_names:
                source_path = os.path.join(from_experiment_directory, 'Configurations', _sweep_file_name)
                target_path = os.path.join(self.configuration_directory, _sweep_file_name)
                if os.path.exists(source_path):
                    shutil.copy(source_path, target_path)
        return super(WandBSweepMixin, self).inherit_configuration(from_experiment_directory, file_name, read)

    def read_config_file(self, file_name='train_config.yml', path=None):
        if self.get_arg('wandb.sweep', False):
            sweep_info_path = os.path.join(self.configuration_directory, 'wandb_sweep_info.yml')
            if os.path.exists(sweep_info_path):
                self.wandb_sweep_id = read_yaml(sweep_info_path)['wandb_sweep_id']
        return super(WandBSweepMixin, self).read_config_file(file_name, path)

    def update_configuration_from_wandb(self, dump_configuration=False):
        if self.wandb_run is None:
            self.initialize_wandb()
        for key in self.wandb_run.config.keys():
            value = self.wandb_run.config[key]
            self.set(key.replace('__', '/'), value)
        if dump_configuration:
            self.dump_configuration()
        return self

    def auto_setup(self, update_git_revision=True, dump_configuration=True):
        super_return = super(WandBSweepMixin, self).auto_setup()
        if self.get_arg('wandb.sweep', False):
            self.update_configuration_from_wandb(dump_configuration=True)
        return super_return


class SweepRunner(BaseExperiment):

    def __init__(self, sweep_experiment_cls, *sweep_experiment_init_args, **sweep_experiment_init_kwargs):
        super(SweepRunner, self).__init__()
        self._wandb_sweep_id = None
        self._sweep_experiment_cls = sweep_experiment_cls
        self._sweep_experiment_init_args = sweep_experiment_init_args
        self._sweep_experiment_init_kwargs = sweep_experiment_init_kwargs
        # Setup
        self.record_args()
        self.read_sweep_info()

    def read_sweep_info(self):
        if self.get_arg('wandb.sweep', False):
            parent_experiment_directory = self.get_arg('inherit', ensure_exists=True)
            # Read stuff from parent experiment
            sweep_info_file_path = os.path.join(parent_experiment_directory, 'Configurations', 'wandb_sweep_info.yml')
            # First try to get sweep_id from commandline args
            if self.get_arg('wandb.sweep_id') is not None:
                self._wandb_sweep_id = self.get_arg('wandb.sweep_id')
            elif os.path.exists(sweep_info_file_path):
                self._wandb_sweep_id = read_yaml(sweep_info_file_path)['wandb_sweep_id']
            else:
                raise RuntimeError("wandb sweep_id could not be found")
        return self

    def make_sweep_function(self, *run_args, **run_kwargs):
        return lambda: self._sweep_experiment_cls(*self._sweep_experiment_init_args,
                                                  **self._sweep_experiment_init_kwargs).run(*run_args, **run_kwargs)

    def run(self, *args, **kwargs):
        if self._wandb_sweep_id is not None and self.get_arg('wandb.sweep', False):
            return wandb.agent(self._wandb_sweep_id, self.make_sweep_function(*args, **kwargs), count=1)
        else:
            return self.make_sweep_function(*args, **kwargs)

