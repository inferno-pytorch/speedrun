import yaml
from yaml.loader import Loader
import operator
from functools import reduce, partial
from copy import deepcopy
from .py_utils import create_instance

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


def hyperopt(loader, node):
    import pdb
    pdb.set_trace()


yaml.add_constructor('!Add', add)
yaml.add_constructor('!Mul', mul)
yaml.add_constructor('!Sub', sub)
yaml.add_constructor('!Div', div)
yaml.add_constructor('!NumpyArray', numpy_array)
yaml.add_constructor('!TorchTensor', torch_tensor)
yaml.add_constructor('!Hyperopt', hyperopt)


object_tags = ['!Obj:', '!Tune:', '!Case:', '!Ref']


class OverrideDict(dict):
    """class to allow overriding of whole dictionaries in recursive_update"""
    def after_override(self):
        return dict(self)


def override_constructor(loader, node):
    if isinstance(node, yaml.MappingNode):
        return OverrideDict(loader.construct_mapping(node))
    else:
        raise NotImplementedError('Node: ' + str(type(node)))


yaml.add_constructor('!Override', override_constructor)


class KeyDeleter:
    """class to allow deletion of dictionarly keys in recursive_update"""
    pass


def key_delete_constructor(loader, node):
    assert node.value == '', f'{node.value}'
    return KeyDeleter()


yaml.add_constructor('!Del', key_delete_constructor)


def _recursive_update_inplace(d1, d2):
    if isinstance(d2, OverrideDict):
        # if requested, just override the whole dict d1
        return d2.after_override()
    for key, value in d2.items():
        if isinstance(value, KeyDeleter):
            # delete the key in d1 if requested
            if key in d1:
                del d1[key]
        elif key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            # if the key is present in both dicts, and both values are dicts, update recursively
            d1[key] = _recursive_update_inplace(d1[key], value)
        else:
            # otherwise, just assign the value
            d1[key] = value
    return d1


def recursive_update(d1, d2):
    """
    Update d1 with the data from d2 recursively
    :param d1: dict
    :param d2: dict
    :return: dict
    """
    # make sure there are no side effects
    d1 = deepcopy(d1)
    d2 = deepcopy(d2)
    return _recursive_update_inplace(d1, d2)


class TempArgObj:
    def __init__(self, value, tag_prefix, tag, style):
        self.value = value
        self.tag_prefix = tag_prefix
        self.tag = tag
        self.style = style

    @staticmethod
    def to_yaml(dumper, data):
        return dumper.represent_scalar(f'{data.tag_prefix}{data.tag}', data.value, style=data.style)


class TempArgsObj(list):
    def __init__(self, value, tag_prefix, tag, flow_style):
        super(TempArgsObj, self).__init__(value)
        self.tag_prefix = tag_prefix
        self.tag = tag
        self.flow_style = flow_style

    @staticmethod
    def to_yaml(dumper, data):
        return dumper.represent_sequence(f'{data.tag_prefix}{data.tag}', data, flow_style=data.flow_style)


class TempKwargsObj(dict):
    def __init__(self, mapping, tag_prefix, tag, flow_style):
        super(TempKwargsObj, self).__init__(mapping)
        # save tag (which is the class to be constructed) in the dict to allow updating
        self['__tag_prefix__'] = tag_prefix
        self['__tag__'] = tag
        self.flow_style = flow_style

    @staticmethod
    def to_yaml(dumper, data):
        tag = data.pop('__tag__')
        tag_prefix = data.pop('__tag_prefix__')
        return dumper.represent_mapping(f'{tag_prefix}{tag}', data, flow_style=data.flow_style)


def temp_obj_constructor(loader, tag_suffix, node, tag_prefix='!Obj'):
    if isinstance(node, yaml.ScalarNode):
        return TempArgObj(loader.construct_scalar(node), tag_prefix, tag_suffix, style=node.style)
    elif isinstance(node, yaml.SequenceNode):
        return TempArgsObj(loader.construct_sequence(node), tag_prefix, tag_suffix, flow_style=node.flow_style)
    elif isinstance(node, yaml.MappingNode):
        return TempKwargsObj(loader.construct_mapping(node), tag_prefix, tag_suffix, flow_style=node.flow_style)
    else:
        raise NotImplementedError('Node: ' + str(type(node)))


for tag_prefix in object_tags:
    yaml.add_multi_constructor(tag_prefix, partial(temp_obj_constructor, tag_prefix=tag_prefix))
yaml.add_representer(TempArgObj, TempArgObj.to_yaml)
yaml.add_representer(TempArgsObj, TempArgsObj.to_yaml)
yaml.add_representer(TempKwargsObj, TempKwargsObj.to_yaml)


class TempKwargsOverrideObj(TempKwargsObj, OverrideDict):
    def after_override(self):
        tag = self.pop('__tag__')
        return TempKwargsObj(mapping=dict(self), tag_prefix='!Obj:', tag=tag, flow_style=self.flow_style)


def temp_override_obj_constructor(loader, tag_suffix, node, tag_prefix='!Obj'):
    if isinstance(node, yaml.ScalarNode):
        return TempArgObj(loader.construct_scalar(node), tag_prefix, tag_suffix, style=node.style)
    elif isinstance(node, yaml.SequenceNode):
        return TempArgsObj(loader.construct_sequence(node), tag_prefix, tag_suffix, flow_style=node.flow_style)
    elif isinstance(node, yaml.MappingNode):
        return TempKwargsOverrideObj(loader.construct_mapping(node), tag_prefix, tag_suffix, flow_style=node.flow_style)
    else:
        raise NotImplementedError('Node: ' + str(type(node)))


yaml.add_multi_constructor('!OverrideObj:', temp_override_obj_constructor)


class TuneLoader(Loader):
    """
    Identical to regular Loader (that does not contruct objects) except in that it constructs Objects marked with the
    "!Tune" tag instead of "!Obj".
    """
    def construct_instance(self, suffix, node):
        if isinstance(node, yaml.MappingNode):  # keyword arguments specified
            class_dict = self.construct_mapping(node, deep=True)
        elif isinstance(node, yaml.SequenceNode):  # positional arguments specified
            class_dict = dict(args=self.construct_sequence(node, deep=True))
        elif isinstance(node, yaml.ScalarNode):  # only one argument specified as scalar
            class_dict = dict(args=[self.construct_scalar(node)])
        else:
            raise NotImplementedError
        return create_instance({suffix: class_dict})


# add the python object constructor to the loader
TuneLoader.add_multi_constructor('!Tune:', TuneLoader.construct_instance)


class CaseLoader(TuneLoader):
    """
    Loader for python object construction
    """

    # we override this method to remember the root node,
    # so that we can later resolve paths relative to it
    def get_single_node(self):
        self.cur_root = super().get_single_node()
        return self.cur_root

    def construct_by_path(self, path):
        cur = self.cur_root
        for item in path.split("."):
            # cur.value, if it's a mappping, contains a list
            # of (key, value) tuples
            for (key, value) in cur.value:
                # key, if it's a scalar, contains its textual
                # content in key.value
                if key.value == item:
                    cur = value
                    break
        # defer construction to the default constructor of
        # the referred node
        result = self.construct_object(cur)
        return result

    def construct_case(self, suffix, node):
        case = self.construct_by_path(suffix)
        case = str(case).lower()
        if isinstance(node, yaml.MappingNode):  # index the mapping with the case
            for (key, value) in node.value:
                # key, if it's a scalar, contains its textual
                # content in key.value
                if str(key.value).lower() == case:
                    return self.construct_object(value)
            from yaml import YAMLError
            raise YAMLError(f'Cannot construct case tag for case "{case}" (read from "{suffix}"). '
                            f'Available cases are {[key.value for key, value in node.value]}.')
        else:
            raise NotImplementedError('!Case is only implemented for mapping nodes')

    def construct_ref(self, node):
        return self.construct_by_path(node.value)



# add the python object constructor to the loader
CaseLoader.add_multi_constructor('!Case:', CaseLoader.construct_case)
CaseLoader.add_constructor('!Ref', CaseLoader.construct_ref)
CaseLoader.add_multi_constructor('!Obj', partial(temp_obj_constructor, tag_prefix='!Obj:'))


# Loader to load !Case and !Ref statements, but not !Obj statements
class ObjectLoader(CaseLoader):
    pass

ObjectLoader.add_multi_constructor('!Obj:', CaseLoader.construct_instance)
