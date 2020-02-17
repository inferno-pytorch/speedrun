import os
import yaml
# from ..core import BaseExperiment
from ..utils.py_utils import flatten_dict
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
