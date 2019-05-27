try:
    import firelight
except ImportError:
    firelight = None

class FirelightMixin(object):

    def _get_visualizer(self, file_name = None):
        # TODO: read from config file other than from train_config.yml ?
        # self.read_config_file(file_name)
        print(self.get('firelight'))
        visualizer = firelight.get_visualizer(self.get('firelight'))
        return visualizer

    def get_image_grid(self, states):
        image_grid = self._get_visualizer(**states)
        return image_grid

