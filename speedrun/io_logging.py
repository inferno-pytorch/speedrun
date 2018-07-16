from .core import BaseExperiment
import os.path as path
import torch
import numpy as np
import imageio


class IOMixin(BaseExperiment):
    @staticmethod
    def to_array(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        elif isinstance(np.ndarray):
            return value
        else:
            raise ValueError(f"Can't convert {value.__class__.__name__} to np.array.")

    def print_image(self, tag, value):
        # Convert to a numpy array
        value = self.to_array(value)
        # Make sure the image axis is right
        if value.ndim == 2:
            # Grayscale
            image = value
        elif value.ndim == 3:
            # RGB or RGBA
            is_correct_shape = value.shape[-1] in [3, 4]
            if not is_correct_shape:
                assert value.shape[0] in [3, 4], "Only RGB and RGBA images are supported."
                image = value.transpose(1, 2, 0)
            else:
                image = value
        else:
            raise ValueError(f"Value must be 2 or 3 dimensional, got {value.ndim} dimensional.")
        # Pick file name
        fields = tag.split('/')
        file_name = f"{fields[-1]}_step_{self.step}.png"
        if len(fields) > 1:
            path_after_plot_dir = path.sep.join(fields[:-1])
        else:
            path_after_plot_dir = ''
        # Make file path
        file_path = path.join(self.plot_directory, path_after_plot_dir, file_name)
        # Write image
        imageio.imwrite(file_path, image)
        # Done
        return self

    def progress(self, iterator, **tqdm_kwargs):
        # TODO
        pass

    def print(self):
        # TODO
        pass
