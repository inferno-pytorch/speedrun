import torch
try:
    import firelight
except ImportError:
    firelight = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class FirelightMixin(object):

    @property
    def visualizer(self):
        if not hasattr(self, '_visualizer'):
            self._visualizer = self.load_visualizer()
        return self._visualizer

    def load_visualizer(self, file_name=None):
        if file_name is not None:
            config = file_name
        else:
            config = self.get('firelight')
        visualizer = firelight.get_visualizer(config)
        return visualizer

    def get_image_grid(self, states):
        image_grid = self.visualizer(**states)
        if isinstance(image_grid, torch.Tensor):
            image_grid.numpy()
        return image_grid

    def save_image_grid(self, states, plot_name='plots'):
        plt.imsave(f'{self.plot_directory}/{plot_name}_step_{self.step}.png', self.get_image_grid(states))
