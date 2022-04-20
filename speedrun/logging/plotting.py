try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import numpy as np
except ImportError:
    np = None


class MatplotlibMixin(object):
    def figure2array(self, figure, channel_first=False, scaling=None):
        assert None not in [plt, np], "figure2array needs matplotlib and numpy to work."
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        buffer = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8)
        buffer.shape = (height, width, 3)
        if channel_first:
            buffer = np.moveaxis(buffer, 2, 0)
        if scaling == "0-1":
            buffer = buffer.astype("float32") / 255.0
        elif scaling == "0-255" or scaling is None:
            pass
        else:
            raise NotImplementedError
        return buffer
