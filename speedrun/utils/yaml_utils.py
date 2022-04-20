import yaml
import operator
from functools import reduce

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None


def add(loader, node):
    return sum(loader.construct_sequence(node))


def mul(loader, node):
    return reduce(operator.mul, loader.construct_sequence(node))


def sub(loader, node):
    return reduce(operator.sub, loader.construct_sequence(node))


def div(loader, node):
    return reduce(operator.truediv, loader.construct_sequence(node))


def numpy_array(loader, node):
    assert np is not None, "Numpy is not importable."
    return np.array(loader.construct_sequence(node))


def torch_tensor(loader, node):
    assert torch is not None, "Torch is not importable."
    return torch.tensor(loader.construct_sequence(node))


yaml.add_constructor("!Add", add)
yaml.add_constructor("!Mul", mul)
yaml.add_constructor("!Sub", sub)
yaml.add_constructor("!Div", div)
yaml.add_constructor("!NumpyArray", numpy_array)
yaml.add_constructor("!TorchTensor", torch_tensor)


def dump_yaml(d, path):
    with open(path, "w+") as f:
        yaml.dump(d, f)
    return d


def read_yaml(path):
    with open(path, "r") as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d
