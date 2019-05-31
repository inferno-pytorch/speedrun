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

    def get_firelight_visualizer(self, file_name=None):
        if file_name is not None:
            config = file_name
        else:
            config = self.get('firelight')
        visualizer = firelight.get_visualizer(config)
        return visualizer

    def get_image_grid(self, states):
        flv = self.get_firelight_visualizer()
        image_grid = flv(**states)
        if isinstance(image_grid, torch.Tensor):
            image_grid.numpy()
        return image_grid

    def save_image_grid(self, states, plot_name='plots'):
        plt.imsave(f'{self.plot_directory}/{plot_name}_step_{self.step}.png', self.get_image_grid(states))
