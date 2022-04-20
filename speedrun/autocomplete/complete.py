from sys import argv

# this updates the default loader
import yaml
import os

# TODO: add completion to --update

working_directory = argv[1]
current_command = argv[2]
current_word_id = int(argv[3])

# ------------- check if config completion should be applied ----------------

current_command = current_command.split()

# print('test_option')

if len(current_command) == 0:
    exit(0)

# get the argument that is currently modified
try:
    current_arg = current_command[current_word_id]
except IndexError:
    exit(0)

if not current_arg.startswith("--config."):
    # not altering arguments in the config
    exit(0)
path_in_config = current_arg.split(".")[1:]

# get the specified config file
try:
    inherit_pos = current_command.index("--inherit")
except ValueError:
    # no --inherit found
    exit(0)
if inherit_pos == len(current_command) - 1:
    # --inherit at the end of the command
    exit(0)

# convert config_path to an absolute path
config_path = os.path.expanduser(current_command[inherit_pos + 1])
if not config_path.startswith("/"):
    config_path = os.path.join(working_directory, config_path)
if not os.path.isfile(config_path):
    # an experiment directory is specified
    config_path = os.path.join(config_path, "Configurations", "train_config.yml")
    if not os.path.isfile(config_path):
        # config file not found
        exit(0)


# ------------- modify yaml loader to only load the nested dict structure ----------------


def none_constructor(*args, **_):
    return None


yaml.add_constructor("!Add", none_constructor)
yaml.add_constructor("!Mul", none_constructor)
yaml.add_constructor("!Sub", none_constructor)
yaml.add_constructor("!Div", none_constructor)
yaml.add_constructor("!NumpyArray", none_constructor)
yaml.add_constructor("!TorchTensor", none_constructor)
yaml.add_constructor("!Hyperopt", none_constructor)


def override_constructor(loader, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    else:
        exit(0)


yaml.add_constructor("!Override", override_constructor)

yaml.add_constructor("!Del", none_constructor)


def override_constructor(loader, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    else:
        # normally, an error is raised
        exit(0)


yaml.add_constructor("!Override", override_constructor)


def temp_obj_constructor(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    else:
        raise NotImplementedError("Node: " + str(type(node)))


yaml.add_multi_constructor("!Obj:", temp_obj_constructor)
yaml.add_multi_constructor("!OverrideObj:", temp_obj_constructor)


class StructureLoader(yaml.Loader):
    """
    Loader for command line auto completion: stops when a non-dictionary is encountered
    Examples:

    """

    def construct_sequence(self, node, deep=False):
        return []


# ------------- load and process the config ----------------

# load the config
try:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=StructureLoader)
except:
    # config could not be loaded for some reason
    exit(0)

# find the current options
for key in path_in_config[:-1]:
    try:
        config = config[key]
    except ValueError:
        exit(0)
if not isinstance(config, dict):
    exit(0)

# print the options to be filtered by bash (maybe filter them here?)
for key in config.keys():
    if str(key).startswith(path_in_config[-1]):
        completion_option = (
            ".".join(["--config"] + path_in_config[:-1]) + "." + str(key)
        )
        if isinstance(config[key], dict):
            # automatically add the dot if there are further nested options
            completion_option += "."
        print(completion_option)
