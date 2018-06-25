from contextlib import contextmanager
import tensorboardX as tx


class TensorboardMixin(object):
    @property
    def logger(self):
        # Build logger if it doesn't exist
        if not hasattr(self, '_logger'):
            # noinspection PyUnresolvedReferences,PyAttributeOutsideInit
            self._logger = tx.SummaryWriter(log_dir=self.log_directory)
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
        return "{}/{}".format(self.tagscope, tag)

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

    def log_embedding(self, tag, tensor, images=None, step=None):
        # noinspection PyUnresolvedReferences
        step = self.step if step is None else step
        self.logger.add_embedding(tag=self.get_full_tag(tag), mat=tensor,
                                  label_img=images, global_step=step)
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