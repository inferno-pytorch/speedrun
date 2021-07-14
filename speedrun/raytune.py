import torch
import ray
from ray import tune

from inferno.trainers.callbacks import Callback

from .core import BaseExperiment
from .tensorboard import TensorboardMixin
from .log_anywhere import register_logger, log_scalar
from .yaml_utils import TuneLoader, ObjectLoader


class DummyExperiment(BaseExperiment, TensorboardMixin):
    DEFAULT_DISPATCH = 'train'
    _name = 'test'

    def __init__(self, config, experiment_directory=None, from_tune=False):
        super(DummyExperiment, self).__init__()
        # Privates
        self._device = None
        self._meta_config['exclude_attrs_from_save'] = ['data_loader', '_device']
        self.is_tune_run = from_tune
        if not self.is_tune_run:
            self.auto_setup()
        else:
            self.record_args()
            if experiment_directory is None:
                self.experiment_directory = tune.get_trial_dir()
            else:
                self.experiment_directory = experiment_directory
            print('Experiment directory:', self.experiment_directory)
            assert isinstance(config, dict), f'{type(config)}'
            self._config = config
            self.dump_configuration()
            self.read_config_file(loader=ObjectLoader)

        # register anywhere logger for scalars
        register_logger(self, 'scalars')

    @classmethod
    def tune(cls, config):
        experiment = cls(config, from_tune=True)
        experiment.run()

    @property
    def log_directory(self):
        """
        Directory where the log files go.
        Overwritten to get a single tensorboard log file with tune.
        """
        if self._experiment_directory is not None:
            return self._experiment_directory
        else:
            return None

    def train(self):
        for epoch in range(3):
            t = torch.randn(1, 3).cuda()
            model = self.get('model')
            model.cuda()
            out = model(t)
            if self.is_tune_run:
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    print('checkpoint_dir:', checkpoint_dir)
            log_scalar('log_anywhere_score', self.get('score'), epoch)
            tune.report(loss=self.get('score') * (1+epoch), accuracy=out)


class TuneMetaExperiment(BaseExperiment):
    DEFAULT_DISPATCH = 'run_trials'

    def __init__(self, experiment_directory=None, trainable=None):
        super().__init__(experiment_directory)
        assert trainable is not None, f'trainable cannot be None'
        self.trainable = trainable
        self.auto_setup(construct_objects=False)
        self.read_config_file(loader=TuneLoader)

    @staticmethod
    def trial_name_string(trial):
        """
        Args:
            trial (Trial): A generated trial object.

        Returns:
            trial_name (str): String representation of Trial.
        """
        return ""

    def run_trials(self):
        tune_config = self.get('tune')
        ray_init_kwargs = tune_config.pop('ray_init', {})
        if tune_config.pop('debug', False):
            import logging
            ray_init_kwargs.update(dict(local_mode=True, num_cpus=1, num_gpus=1, logging_level=logging.DEBUG))
        ray.init(**ray_init_kwargs)
        result = ray.tune.run(
            self.trainable,
            config=self.get('trial'),
            local_dir=self.experiment_directory,
            name="Trials",
            trial_name_creator=self.trial_name_string,
            **tune_config
        )
        print("Best hyperparameters found were: ", result.best_config)


class TuneReportingCallback(Callback):
    def end_of_validation_run(self, **_):
        self.trainer.get_state('validation_score')
        # TODO: also keep track of and log average training scores
        validation_loss = self.trainer.get_state('validation_loss_averaged')
        validation_error = self.trainer.get_state('validation_error_averaged')
        tune.report(validation_error=validation_error, validation_loss=validation_loss)
