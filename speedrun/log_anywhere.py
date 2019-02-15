_LOGGERS = []


def register_logger(logger):
    _LOGGERS.append(logger)


def _log_object(object_type, *args, **kwargs):
    log_func_name = 'log_' + object_type
    for logger in _LOGGERS:
        if hasattr(logger, log_func_name):
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
