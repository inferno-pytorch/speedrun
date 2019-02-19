_LOGGERS = []
_KEYS = []

_KEY_MAPPING = {
    'scalar': 'scalar',
    'scalars': 'scalar',
    'image': 'image',
    'images': 'image',
    'embedding': 'embedding',
    'embeddings': 'embedding',
    'text': 'text',
    'histogram': 'histogram',
    'histograms': 'histogram',
}


def register_logger(logger, keys):
    if keys in (False, None):
        return
    _LOGGERS.append(logger)
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(keys, (list, tuple)):
        for key in keys:
            assert key in _KEY_MAPPING, f'Key {key} not found. Available keys: {list(_KEY_MAPPING.keys())}'
        keys = [_KEY_MAPPING[key] for key in keys]
    _KEYS.append(keys)


def _log_object(object_type, *args, **kwargs):
    log_func_name = 'log_' + object_type
    for logger, keys in zip(_LOGGERS, _KEYS):
        if keys not in (True, 'all'):
            use_current_logger = log_func_name in keys
        else:
            use_current_logger = hasattr(logger, log_func_name)
        if use_current_logger:
            logger.__getattribute__(log_func_name)(*args, **kwargs)


def log_scalar(*args, **kwargs):
    _log_object('scalar', *args, **kwargs)


def log_image(*args, **kwargs):
    _log_object('image', *args, **kwargs)


def log_embedding(*args, **kwargs):
    _log_object('embedding', *args, **kwargs)


def log_text(*args, **kwargs):
    _log_object('text', *args, **kwargs)


def log_histogram(*args, **kwargs):
    _log_object('histogram', *args, **kwargs)
