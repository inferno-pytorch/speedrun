from contextlib import contextmanager
import torch

try:
    import inferno
except ImportError:
    inferno = None
try:
    from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
except ImportError:
    TensorboardLogger = None

try:
    from firelight.inferno_callback import get_visualization_callback as firelight_visualizer
except ImportError:
    firelight_visualizer = None

class InfernoMixin(object):

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
            self._trainer = inferno.trainers.basic.Trainer(self.model)\
                                   .save_to_directory(self.experiment_directory)

            # call all defined bind functions
            for fname in dir(self):
                if fname.startswith('inferno_build_'):
                    getattr(self, fname)()

            self._trainer.to(self.device)

        return self._trainer

    def build_model(self):
        raise NotImplementedError("Overwrite this function to specify model")

    @property
    def model(self):
        # Build model if it doesn't exist
        if not hasattr(self, '_model'):
            self._model = self.build_model()
        return self._model

    def inferno_build_criterion(self):
        print("Using criterion ", self.get('trainer/criterion'))
        self._trainer.build_criterion(self.get('trainer/criterion'))

    def inferno_build_metric(self):
        if self.get('trainer/metric') is not None:
            self._trainer.build_metric(self.get('trainer/metric'))
        else:
            print("No metric specified")

    def inferno_build_optimizer(self):
        print("Building optimizer")
        self._trainer.build_optimizer(self.get('trainer/optimizer'),
                                      **self.get('trainer/optimizer_kwargs'))

    def inferno_build_intervals(self):
        if self.get('trainer/intervals/validate_every') is not None:
            self._trainer.validate_every(**self.get('trainer/intervals/validate_every'))

        if self.get('trainer/intervals/save_every') is not None:
            self._trainer.save_every(self.get('trainer/intervals/save_every'))

    def inferno_build_tensorboard(self):
        if self.get('trainer/tensorboard') is not None:
            if TensorboardLogger is None:
                print("warning can not use TensorboardLogger")
                return

            tb_args = self.get('trainer/tensorboard')
            tb_args['log_directory'] = f"{self.experiment_directory}/Logs"
            print("logging to ", tb_args['log_directory'])
            tb_logger = TensorboardLogger(**tb_args)

            # register Tensorboard logger
            self._trainer.build_logger(tb_logger)
            # and set _logger to so it can be used by the Tensorboardmixin
            self._logger = tb_logger

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
                cb_class_module = getattr(inferno.trainers.callbacks, cb_class)
                for cb in self.get(f'trainer/callbacks/{cb_class}'):
                    print(f'creating trainer/callbacks/{cb_class}/{cb}')
                    args = self.get(f'trainer/callbacks/{cb_class}/{cb}')
                    if "noargs" in args:
                        callback = getattr(cb_class_module, cb)()
                    else:
                        callback = getattr(cb_class_module, cb)(**args)
                    self._trainer.register_callback(callback)

        if self.get('firelight') is not None:
            print(self.get('firelight'))
            if firelight_visualizer is None:
                raise ImportError("firelight could not be imported but is present in the config file")
            else:
                flc = firelight_visualizer(self.get('firelight'))
                self._trainer.register_callback(flc)

    # overwrite this function to define train loader
    def build_train_loader(self):
        raise NotImplementedError()

    # overwrite this function to define validation loader
    def build_val_loader(self):
        return None

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

    def inferno_build_loaders(self):
        self._trainer.bind_loader('train', self.train_loader) \

        if self.val_loader is not None:
            self._trainer.bind_loader('validate', self.val_loader)

    def train(self):
        return self.trainer.fit()
