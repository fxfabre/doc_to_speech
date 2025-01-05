import logging
from functools import wraps
from time import time

logger = logging.getLogger(__name__)


def timeit(method):
    @wraps(method)
    def timed(*args, **kw):
        start = time()
        result = method(*args, **kw)

        logging.info("%r run in %.2f sec", method.__name__, time() - start)
        return result

    return timed
