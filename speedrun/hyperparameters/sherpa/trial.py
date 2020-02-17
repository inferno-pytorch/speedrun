import os

from ...core import register_pre_dispatch_hook, BaseExperiment
try:
    from sherpa import Client
except ImportError:
    Client = None


class SherpaTrialMixin(BaseExperiment):
    @property
    def sherpa_client(self):
        if not hasattr(self, '_sherpa_client'):
            assert Client is not None, "sherpa could not be imported. Is it installed?"
            # noinspection PyAttributeOutsideInit
            self._sherpa_client = Client()
        return self._sherpa_client

    @property
    def sherpa_trial(self):
        if not hasattr(self, '_sherpa_trial'):
            # noinspection PyAttributeOutsideInit
            self._sherpa_trial = self.sherpa_client.get_trial()
        return self._sherpa_trial

    def sherpa_report(self, objective, iteration='epochs', **extra_metrics):
        if iteration in ['epochs', 'epoch']:
            iteration = self.epoch
        elif iteration in ['steps', 'step']:
            iteration = self.step
        else:
            assert isinstance(iteration, int), f"Iteration must be an integer, got {type(iteration)} instead."
        self.sherpa_client.send_metrics(trial=self.sherpa_trial, iteration=iteration,
                                        objective=objective, context=extra_metrics)
        return self

    def parse_experiment_directory(self):
        directory, directory_name = os.path.split(self.get_arg(1))
        if directory_name == 'SPEEDRUN-SET-TRIAL-ID':
            self.experiment_directory = os.path.join(directory, f'TRIAL-{self.sherpa_trial.id}')
        else:
            BaseExperiment.parse_experiment_directory(self)
        return self

    def update_configuration_from_args(self):
        if Client is None:
            # No sherpa, fallback to the speedrun behaviour
            return BaseExperiment.update_configuration_from_args(self)
        # Pretend sherpa trial parameters are args
        for name, value in self.sherpa_trial.parameters.items():
            self.set(name, value)
        return self

