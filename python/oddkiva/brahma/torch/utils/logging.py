import logging

from oddkiva.brahma.torch.parallel.ddp import (
    get_local_rank,
    get_rank
)


def logd(logger: logging.Logger, message: str):
    local_rank = get_local_rank()
    rank = get_rank()
    msg = f"[R:{rank},LR:{local_rank}]  {message}"
    logger.debug(msg)


def logi(logger: logging.Logger, message: str):
    local_rank = get_local_rank()
    rank = get_rank()
    msg = f"[R:{rank},LR:{local_rank}]  {message}"
    logger.info(msg)


def logw(logger: logging.Logger, message: str):
    local_rank = get_local_rank()
    rank = get_rank()
    msg = f"[R:{rank},LR:{local_rank}]  {message}"
    logger.warning(msg)
