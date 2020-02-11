from ..core import register_pre_dispatch_hook
try:
    from sherpa import Client
except ImportError:
    Client = None


class SherpaTrialMixin(object):
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

    @register_pre_dispatch_hook
    def update_configuration_with_sherpa_trial(self):
        if Client is None:
            # No sherpa :-(
            return
        for name, value in self.sherpa_trial.parameters.items():
            self.set(name, value)
        self.dump_configuration()

