from loguru import logger
from oddkiva.brahma.torch.utils.logging import format_msg


def test_log():
    logger.debug(format_msg("Hello, World!"))
    logger.info(format_msg("Hello, World!"))
    logger.warning(format_msg("Hello, World!"))
