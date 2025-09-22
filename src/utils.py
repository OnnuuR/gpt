\
import logging, time, functools, random

def get_logger(name="btc_machine", level="INFO"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO))
    return logger

def retry(exceptions, tries=5, delay=1.0, backoff=2.0, jitter=0.2):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    time.sleep(_delay + random.random() * jitter)
                    _tries -= 1
                    _delay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return deco
