from contextlib import contextmanager
try:
    import tensorboardX as tx
except ImportError:
    tx = None


class TensorboardMixin(object):
    @property
    def logger(self):
        if tx is None:
            raise ModuleNotFoundError("TensorboardMixin requires tensorboardX. You can "
                                      "install it with `pip install tensorboardX`")
        # Build logger if it doesn't exist
        if not hasattr(self, '_logger'):
            # noinspection PyUnresolvedReferences,PyAttributeOutsideInit
            self._logger = tx.SummaryWriter(logdir=self.log_directory)
            # noinspection PyUnresolvedReferences
            self._meta_config['exclude_attrs_from_save'].append('_logger')
        return self._logger

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

    def log_scalar(self, tag, value, step=None):
        # noinspection PyUnresolvedReferences
        step = self.step if step is None else step
        self.logger.add_scalar(tag=self.get_full_tag(tag), scalar_value=value,
                               global_step=step)
        return self

    def log_image(self, tag, value, step=None):
        # noinspection PyUnresolvedReferences
        step = self.step if step is None else step
        self.logger.add_image(tag=self.get_full_tag(tag), img_tensor=value,
                              global_step=step)
        return self

    def log_embedding(self, tag, tensor, images=None, metadata=None, step=None):
        # noinspection PyUnresolvedReferences
        step = self.step if step is None else step
        self.logger.add_embedding(tag=self.get_full_tag(tag), mat=tensor,
                                  metadata=metadata, label_img=images, global_step=step)
        return self

    def log_histogram(self, tag, value, bins='tensorflow', step=None):
        # noinspection PyUnresolvedReferences
        step = self.step if step is None else step
        self.logger.add_histogram(tag=self.get_full_tag(tag), values=value, bins=bins,
                                  global_step=step)
        return self

    def _log_x_now(self, x):
        # noinspection PyUnresolvedReferences
        frequency = self.get(f'tensorboard/log_{x}_every', None)
        if frequency is not None:
            # noinspection PyUnresolvedReferences
            return (self.step % frequency) == 0
        else:
            return False

    @property
    def log_scalars_now(self):
        return self._log_x_now('scalars')

    @property
    def log_images_now(self):
        return self._log_x_now('images')

    @property
    def log_embeddings_now(self):
        return self._log_x_now('embeddings')

    @property
    def log_histograms_now(self):
        return self._log_x_now('histograms')

