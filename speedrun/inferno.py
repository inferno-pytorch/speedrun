from contextlib import contextmanager
import torch
from torch.utils.data import DataLoader
import os
from .py_utils import locate, get_single_key_value_pair, create_instance
from .log_anywhere import register_logger, log_scalar
from .tensorboard import TensorboardMixin
from inferno.io.transform import Compose

try:
    import inferno
except ImportError:
    inferno = None
try:
    from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
except ImportError:
    TensorboardLogger = None

try:
    from inferno.trainers.callbacks.base import Callback
except ImportError:
    Callback = object

try:
    from firelight.inferno_callback import get_visualization_callback as firelight_visualizer
except ImportError:
    firelight_visualizer = None

# logger to sent images to the trainer to be visualized by firelight from anywhere


class FirelightLogger(object):

    def __init__(self, trainer):
        self.trainer = trainer

    def log_image(self, tag, value):
        self.trainer.update_state(tag, value.detach().cpu())


class IncreaseStepCallback(Callback):
    def __init__(self, experiment):
        super(IncreaseStepCallback, self).__init__()
        self.experiment = experiment

    def end_of_training_iteration(self, **_):
        print('jep')
        self.experiment.next_step()
        log_scalar('step', self.experiment.step)

    def __getstate__(self):
        return {}


class ParsingMixin(object):
    """
    The ParsingMixin provides a convenient way to create
    model, criterion metric and data loaders from
    You may overwrite the build methods in the derived experiment classes
    to define the model, criterion and metric that are harder to instantiate
    (e.g. by requiring specially constructed input objects)
    Note, that these methods should return the object in question,
    which makes it possible to use the parent objects
    """

    def build_model(self):
        model_dict = self.get('model')
        
        model_path = model_dict[next(iter(model_dict.keys()))].pop('loadfrom', None)
        model = create_instance(model_dict)
        
        if model_path is not None:
            print(f"loading model from {model_path}")
            state_dict = torch.load(model_path)["_model"].state_dict()
            model.load_state_dict(state_dict)

        return model

    @property
    def model(self):
        # Build model if it doesn't exist
        if not hasattr(self, '_model'):
            # noinspection PyAttributeOutsideInit
            self._model = self.build_model()
            assert self._model is not None
        return self._model

    def build_train_loader(self):
        loader_kwargs = self.get('loader')
        dataset = create_instance(loader_kwargs['dataset'])
        loader_kwargs['dataset'] = dataset
        return DataLoader(**loader_kwargs)

    # overwrite this function to define validation loader
    def build_val_loader(self):
        loader_kwargs = self.get('val_loader')
        dataset = create_instance(loader_kwargs['dataset'])
        loader_kwargs['dataset'] = dataset

        return DataLoader(**loader_kwargs)

    def build_criterion(self):
        return create_instance(self.get('criterion'))

    def build_metric(self):
        metric = self.get('metric')
        if metric is not None:
            metric = create_instance(metric)
        return metric

    @property
    def criterion(self):
        if not hasattr(self, '_criterion'):
            # noinspection PyAttributeOutsideInit
            self._criterion = self.build_criterion()
        return self._criterion

    @property
    def metric(self):
        if not hasattr(self, '_metric'):
            # noinspection PyAttributeOutsideInit
            self._metric = self.build_metric()
        return self._metric


class InfernoMixin(ParsingMixin):

    @property
    def tagscope(self):
        if not hasattr(self, '_tagscope'):
            # noinspection PyAttributeOutsideInit
            self._tagscope = ''
        return self._tagscope

    @contextmanager
    def set_tagscope(self, name):
        try:
            self._tagscope = name
            yield
        finally:
            # noinspection PyAttributeOutsideInit
            self._tagscope = ''

    def get_full_tag(self, tag):
        if self.tagscope:
            return "{}/{}".format(self.tagscope, tag)
        else:
            return tag

    @property
    def device(self):
        if self._device is None:
            # noinspection PyAttributeOutsideInit
            self._device = torch.device(self.get('device'))
        return self._device

    @property
    def trainer(self):
        """
        inferno trainer. Will be constructed on first use.
        """
        if inferno is None:
            raise ModuleNotFoundError("InfernoMixin requires inferno. You can "
                                      "install it with `pip install in "
                                      "pip install inferno-pytorch`")
        # Build trainer if it doesn't exist
        if not hasattr(self, '_trainer'):
            # noinspection PyAttributeOutsideInit
            self._trainer = inferno.trainers.basic.Trainer(self.model)\
                                   .save_to_directory(self.experiment_directory)

            # call all defined bind functions
            for fname in dir(self):
                if fname.startswith('inferno_build_'):
                    getattr(self, fname)()

            # add callback to increase step counter
            # noinspection PyUnresolvedReferences
            self._trainer.register_callback(IncreaseStepCallback(self))

            self._trainer.to(self.device)

        return self._trainer

    def inferno_build_criterion(self):
        self._trainer.build_criterion(self.criterion)

    def inferno_build_metric(self):
        if self.metric is not None:
            self._trainer.build_metric(self.metric)
        else:
            print("No metric specified")

    def inferno_build_optimizer(self):
        optimizer_dict = self.get('trainer/optimizer')
        for o in optimizer_dict:
            self._trainer.build_optimizer(o, **optimizer_dict[o])

    _INTERVAL_KEYS = ['validate_every', 'save_every', 'evaluate_metric_every']

    def inferno_build_intervals(self):
        for key, arguments in self.get('trainer/intervals').items():
            assert key in self._INTERVAL_KEYS, f'Cannot set interval "{key}". Valid intervals: {self._INTERVAL_KEYS}'
            if isinstance(arguments, dict):
                getattr(self._trainer, key)(**arguments)
            else:
                getattr(self._trainer, key)(arguments)

    def inferno_build_tensorboard(self):
        if self.get('trainer/tensorboard') is not None:
            if TensorboardLogger is None:
                print("warning can not use TensorboardLogger")
                return

            tb_args = self.get('trainer/tensorboard')

            # pop arguments specifying config logging
            log_config = tb_args.pop('log_config', True)
            split_keys = tb_args.pop('split_config_keys', True)

            # pop argument specifying anywhere logging for TensorboardMixin # TODO: how to put this in TensorboardMixin?
            log_anywhere_keys = tb_args.pop('log_anywhere', None)

            tb_args['log_directory'] = f"{self.experiment_directory}/Logs"
            print("logging to ", tb_args['log_directory'])
            tb_logger = TensorboardLogger(**tb_args)

            # register Tensorboard logger
            self._trainer.build_logger(tb_logger)
            # and set _logger to so it can be used by the TensorboardMixin
            self._logger = tb_logger.writer
            if log_config:
                self.log_configuration(split_keys)
            # If experiment also uses TensorboardMixin, register it for anywhere logging
            if isinstance(self, TensorboardMixin):
                register_logger(self, log_anywhere_keys)
            else:
                assert log_anywhere_keys is None, f'Cannot register anywhere logging for keys {log_anywhere_keys}, ' \
                                                  f'please inherit from TensorboardMixin.'

    def log_configuration(self, split_keys=True):
        for filename in os.listdir(self.configuration_directory):
            print(f'logging {filename}')
            if filename.endswith('.yml'):
                with open(os.path.join(self.configuration_directory, filename)) as f:
                    if split_keys:
                        tags = []
                        paragraphs = []
                        for i, line in enumerate(f.readlines()):
                            # add tab to each line to make sure the paragraph is formatted as code
                            if not line.startswith((' ', '#', '\t')) and len(line.split(':')) > 1:
                                paragraphs.append('\t' + ':'.join(line.split(':')[1:]))
                                tags.append(line.split(':')[0])
                            else:
                                paragraphs[-1] += '\t' + line
                        for tag, paragraph in zip(tags, paragraphs):
                            self._logger.add_text(tag=self.get_full_tag('/'.join([tag, filename])),
                                                  text_string=paragraph, global_step=0)
                    else:
                        text = '\t' + f.read().replace('\n', '\n\t')
                        self._logger.add_text(tag=self.get_full_tag(filename), text_string=text,
                                              global_step=0)

    def inferno_build_limits(self):
        if self.get(f'trainer/max_epochs') is not None:
            self._trainer.set_max_num_epochs(self.get(f'trainer/max_epochs'))
        elif self.get(f'trainer/max_iterations') is not None:
            self._trainer.set_max_num_iterations(self.get(f'trainer/max_iterations'))
        else:
            print("No termination point specified!")

    def inferno_build_callbacks(self):
        # build all callbacks from nested conf file
        if self.get('trainer/callbacks') is not None:
            for cb_class in self.get('trainer/callbacks'):
                cb_class_module = locate(cb_class, inferno.trainers.callbacks)
                for cb in self.get(f'trainer/callbacks/{cb_class}'):
                    print(f'creating trainer/callbacks/{cb_class}/{cb}')
                    args = self.get(f'trainer/callbacks/{cb_class}/{cb}')
                    if "noargs" in args:
                        callback = getattr(cb_class_module, cb)()
                    else:
                        callback = getattr(cb_class_module, cb)(**args)
                    self._trainer.register_callback(callback)

        if self.get('firelight') is not None:
            if firelight_visualizer is None:
                raise ImportError("firelight could not be imported but is present in the config file")
            else:
                # if requested, register anywhere logger for firelight
                register_logger(FirelightLogger(self.trainer), self.get('firelight').pop('log_anywhere', 'all'))

                flc = firelight_visualizer(self.get('firelight'))
                self._trainer.register_callback(flc)

    @property
    def train_loader(self):
        # Build model if it doesn't exist
        if not hasattr(self, '_train_loader'):
            self._train_loader = self.build_train_loader()
        return self._train_loader

    @property
    def val_loader(self):
        if not hasattr(self, '_val_loader'):
            self._val_loader = self.build_val_loader()
        return self._val_loader

    @property
    def num_targets(self):
        return self.get('trainer/num_targets', 1)

    def inferno_build_loaders(self):
        self._trainer.bind_loader('train',
                                  self.train_loader,
                                  num_targets=self.num_targets)

        if self.val_loader is not None:
            self._trainer.bind_loader('validate',
                                      self.val_loader,
                                      num_targets=self.num_targets)

    def create_transform(self, list_of_transforms):
        return Compose(*[create_instance(t) for t in list_of_transforms])

    def train(self):
        return self.trainer.fit()
