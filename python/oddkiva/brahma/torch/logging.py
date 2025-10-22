import logging

from oddkiva.brahma.torch.parallel.ddp import (
    get_local_rank, 
    get_rank
)


FORMAT = '%(asctime)s %(rank)s %(local_rank)s %(message)s'


def logd(logger: logging.Logger, message: str):
    extra = {
        'local_rank': get_local_rank(),
        'rank': get_rank()
    }
    logger.debug(message, extra=extra)


def logi(logger: logging.Logger, message: str):
    extra = {
        'local_rank': get_local_rank(),
        'rank': get_rank()
    }
    logger.info(message, extra=extra)


def logw(logger: logging.Logger, message: str):
    extra = {
        'local_rank': get_local_rank(),
        'rank': get_rank()
    }
    logger.warning(message, extra=extra)
