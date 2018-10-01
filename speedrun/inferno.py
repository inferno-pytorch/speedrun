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

            # self._meta_config['exclude_attrs_from_save'].append('_logger')
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
        self._trainer.build_optimizer(self.get('trainer/optimizer'))

    def inferno_build_intervals(self):
        print(self.get('trainer/intervals'))
        if self.get('trainer/intervals/validate_every') is not None:
            self._trainer.validate_every(self.get('trainer/intervals/validate_every'))

        if self.get('trainer/intervals/save_every') is not None:
            self._trainer.save_every(self.get('trainer/intervals/save_every'))

    def inferno_build_tensorboard(self):
        if self.get('trainer/tensorboard') is not None:
            if TensorboardLogger is None:
                print("warning can not use TensorboardLogger")
                return

            tb_logger = TensorboardLogger(**self.get(f'trainer/tensorboard/'))
            # register tensorboard logger
            self._trainer.build_logger(tb_logger)
            # and set _logger to so it can be used by the tensorboard mixin
            self._logger = tb_logger

    def inferno_build_max_epochs(self):
        if 'max_epochs' in self.get(f'trainer'):
            self._trainer.set_max_num_epochs(self.get(f'trainer/max_epochs'))

    # overwrite this properties to define train loader
    def build_train_loader(self):
        raise NotImplementedError()

    # overwrite this properties to define validation loader
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
