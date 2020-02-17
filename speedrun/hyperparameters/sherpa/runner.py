import os
import logging
import datetime
import shlex

from sherpa.core import Parameter, _port_finder, logger, Study, _Database, _Runner
import sherpa.algorithms as shalg
import sherpa.schedulers as shsch

try:
    # Try to get the raven scheduler if raven is available
    from raven.sherpa.scheduler import RavenScheduler
except ImportError:
    RavenScheduler = None

from ...core import BaseExperiment


def optimize(parameters, algorithm, lower_is_better,
             scheduler,
             command=None,
             filename=None,
             output_dir=None,
             max_concurrent=1,
             db_port=None, stopping_rule=None,
             dashboard_port=None, resubmit_failed_trials=False, verbose=1,
             load=False, mongodb_args={}, disable_dashboard=False):
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser('~'), '.sherpa',
                                  str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not scheduler.output_dir:
        scheduler.output_dir = output_dir

    if verbose == 0:
        logger.setLevel(level=logging.INFO)
        logging.getLogger('dblogger').setLevel(level=logging.WARNING)

    study = Study(parameters=parameters,
                  algorithm=algorithm,
                  lower_is_better=lower_is_better,
                  stopping_rule=stopping_rule,
                  dashboard_port=dashboard_port,
                  output_dir=output_dir,
                  disable_dashboard=disable_dashboard)

    if command:
        runner_command = shlex.split(command)
    elif filename:
        runner_command = ['python', filename]
    else:
        raise ValueError("Need to provide either command or filename.")

    if load:
        study.load()

    if not db_port:
        db_port = _port_finder(27001, 27050)

    with _Database(db_dir=output_dir, port=db_port,
                   reinstantiated=load, mongodb_args=mongodb_args) as db:
        runner = _Runner(study=study,
                         scheduler=scheduler,
                         database=db,
                         max_concurrent=max_concurrent,
                         command=runner_command,
                         resubmit_failed_trials=resubmit_failed_trials)
        runner.run_loop()
    return study.get_best_result()


class SherpaLauncher(BaseExperiment):
    """
    Target API
    ----------
    python -m raven.speedrun.sherpa exp/SWEEP-1 --inherit exp/SWEEP-0 --script train.py --base_config exp/EXP-0
    """
    def __init__(self, auto_setup=False, build=False):
        super(SherpaLauncher, self).__init__()
        # Privates
        self._parameters = None
        self._algorithm = None
        self._scheduler = None
        self._launch_command = None
        # Publics
        if auto_setup:
            self.auto_setup()
        if build:
            self.build()

    @BaseExperiment.experiment_directory.setter
    def experiment_directory(self, value):
        super(SherpaLauncher, type(self)).experiment_directory.fset(self, value)
        if value is not None:
            # Add an extra directory to store the trials in
            os.makedirs(os.path.join(value, 'Trials'), exist_ok=True)

    @property
    def trial_directory(self):
        if self._experiment_directory is not None:
            return os.path.join(self._experiment_directory, 'Trials')
        else:
            return None

    def build(self):
        self.build_parameters()
        self.build_algorithm()
        self.build_scheduler()
        self.build_launch_command()
        return self

    def build_parameters(self):
        if self.get('parameter_grid', None) is not None:
            assert self.get('parameters', None) is None, "Configuration specifies both `parameter_grid` and" \
                                                         " `parameters`. It may only specify one at a time."
            self._parameters = Parameter.grid(self.get('parameter_grid'))
        else:
            assert self.get('parameters', None) is not None, "Neither `parameter_grid` nor `parameters` were specified."
            self._parameters = [Parameter.from_dict(p_dict) for p_dict in self.get('parameters')]
        return self

    @property
    def parameters(self):
        if self._parameters is None:
            self.build_parameters()
        return self._parameters

    def build_algorithm(self):
        self._algorithm = \
            getattr(shalg, self.get('algorithm/name', ensure_exists=True))(**self.get('algorithm/kwargs', {}))
        return self

    @property
    def algorithm(self):
        if self._algorithm is None:
            self.build_algorithm()
        return self._algorithm

    def build_scheduler(self):
        scheduler_name = self.get('scheduler/name', 'local')
        if scheduler_name == 'RavenScheduler':
            # Special case for raven schedulers
            assert RavenScheduler is not None
            # noinspection PyCallingNonCallable
            self._scheduler = RavenScheduler(output_dir=self.log_directory, **self.get('scheduler/kwargs', {}))
        else:
            self._scheduler = \
                getattr(shsch, scheduler_name)(output_dir=self.log_directory, **self.get('scheduler/kwargs', {}))
        return self

    @property
    def scheduler(self):
        if self._scheduler is None:
            self.build_scheduler()
        return self._scheduler

    def build_launch_command(self):
        # The trial mixin should be able to understand these.
        launch_command_components = [
            'python',
            self.get_arg('script', ensure_exists=True),
            os.path.join(self.trial_directory, 'SPEEDRUN-SET-TRIAL-ID'),
            '--inherit',
            self.get_arg('base_config', ensure_exists=True),
        ]
        self._launch_command = ' '.join(launch_command_components)
        return self

    @property
    def launch_command(self):
        if self._launch_command is None:
            self.build_launch_command()
        return self._launch_command

    def launch(self):
        result = optimize(parameters=self.parameters, algorithm=self.algorithm, command=self.launch_command,
                          scheduler=self.scheduler, output_dir=self.log_directory,
                          lower_is_better=self.get('general/lower_is_better', True),
                          max_concurrent=self.get('general/max_concurrent_jobs', 1),
                          db_port=self.get('general/db_port'), mongodb_args=self.get('general/mongodb_args', {}),
                          stopping_rule=self.stopping_rule, load=self.get_arg('load_study', False),
                          resubmit_failed_trials=self.get('general/resubmit_failed_trials', False),
                          disable_dashboard=self.get_arg('disable_dashboard', False),
                          dashboard_port=self.get('general/dashboard_port'))
        return result


if __name__ == '__main__':
    SherpaLauncher(auto_setup=True, build=True).launch()