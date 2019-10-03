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
    if isinstance(keys, (list, tuple)):
        for key in keys:
            assert key in _KEY_MAPPING, f'Key {key} not found. Available keys: {list(_KEY_MAPPING.keys())}'
        keys = [_KEY_MAPPING[key] for key in keys]
    _KEYS.append(keys)


ALLOW_NO_MATCH = True


def _log_object(object_type, *args, allow_no_match=None, **kwargs):
    allow_no_match = ALLOW_NO_MATCH if allow_no_match is None else allow_no_match
    log_func_name = 'log_' + object_type
    logger_matched = False
    for logger, keys in zip(_LOGGERS, _KEYS):
        if keys not in (True, 'all'):
            use_current_logger = object_type in keys
        else:
            use_current_logger = hasattr(logger, log_func_name)
        if use_current_logger:
            logger_matched = True
            logger.__getattribute__(log_func_name)(*args, **kwargs)
    assert allow_no_match or logger_matched, f'No logger to log "{object_type}" registered.'


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
