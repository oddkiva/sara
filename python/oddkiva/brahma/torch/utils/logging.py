import logging

from oddkiva.brahma.torch.parallel.ddp import (
    get_local_rank,
    get_rank
)


def format_msg(msg: str) -> str:
    local_rank = get_local_rank()
    rank = get_rank()
    if local_rank is not None and rank is not None:
        return f"[R:{rank},LR:{local_rank}]  {msg}"
    else:
        return msg


def logd(logger: logging.Logger, msg: str):
    logger.debug(format_msg(msg))


def logi(logger: logging.Logger, msg: str):
    logger.info(format_msg(msg))


def logw(logger: logging.Logger, msg: str):
    logger.warning(format_msg(msg))
