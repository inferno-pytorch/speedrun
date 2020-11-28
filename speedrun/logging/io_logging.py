import os.path as path
import os

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import imageio
except ImportError:
    imageio = None

try:
    import tqdm
except ImportError:
    tqdm = None

from ..utils.yaml_utils import dump_yaml


class IOMixin(object):
    @property
    def is_printing_to_file(self):
        return getattr(self, '_print_to_file', False)

    @property
    def printing_to_file_name(self):
        return getattr(self, '_print_filename', 'stdout')

    def print_to_file(self, yes=True, fname='stdout'):
        setattr(self, '_print_to_file', yes)
        setattr(self, '_print_filename', fname)
        return self

    @property
    def printer(self):
        return getattr(self, '_printer', print)

    def set_printer(self, printer):
        if printer == 'stdout':
            setattr(self, '_printer', print)
        elif printer == 'tqdm':
            setattr(self, '_printer', tqdm.tqdm.write)
        else:
            setattr(self, '_printer', printer)

    def print_to_tqdm(self):
        assert tqdm is not None, "tqdm is required to print_to_tqdm. Please `pip install tqdm`."
        self.set_printer('tqdm')

    @staticmethod
    def to_array(value):
        assert np is not None, "numpy is required for checking if value is numpy array (surprise!)."
        if torch is not None and torch.is_tensor(value):
            return value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            return value
        else:
            raise ValueError(f"Can't convert {value.__class__.__name__} to np.array.")

    # noinspection PyUnresolvedReferences
    def print_image(self, tag, value):
        assert imageio is not None, "imageio is required to print images."
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
                image = value.transpose((1, 2, 0))
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
        # Make directory if it doesn't exist
        os.makedirs(path.join(self.plot_directory, path_after_plot_dir), exist_ok=True)
        # Make file path
        file_path = path.join(self.plot_directory, path_after_plot_dir, file_name)
        # Write image
        imageio.imwrite(file_path, image)
        # Done
        return self

    @property
    def progress_bars(self):
        # Make a dict of progress bars if not available
        if not hasattr(self, '_progress_bars'):
            setattr(self, '_progress_bars', {})
        return getattr(self, '_progress_bars')

    def progress(self, iterator, tag=None, **tqdm_kwargs):
        assert tqdm is not None, "tqdm is required for progress bars. Please `pip install tqdm`."
        progress_bar = tqdm.tqdm(iterator, **tqdm_kwargs)
        if tag is not None:
            self.progress_bars[tag] = progress_bar
        return progress_bar

    def log_progress(self, tag, **items):
        progress_bar = self.progress_bars.get(tag)
        if progress_bar is None:
            pass
        else:
            progress_bar.set_postfix(**items)
        return self

    # noinspection PyUnresolvedReferences
    def print(self, message, printer=None):
        if not printer:
            printer = self.printer
        if printer == 'tqdm':
            printer = tqdm.tqdm.write
        # Print to std-out with printer
        printer(message)
        if self.is_printing_to_file:
            # Write message out to file
            stdout_filename = path.join(self.log_directory, self.printing_to_file_name)
            with open(stdout_filename, 'a') as stdout_file:
                print(message, end='\n', file=stdout_file)
        return self

    @property
    def yaml_dump_directory(self):
        yaml_dump_directory = path.join(self.log_directory, "YAMLDumps")
        if not os.path.exists(yaml_dump_directory):
            os.makedirs(yaml_dump_directory, exist_ok=True)
        return yaml_dump_directory

    def dump_yaml(self, tag, payload):
        dump_yaml(payload, path=os.path.join(self.yaml_dump_directory, f"{tag}.yml"))
        return self
