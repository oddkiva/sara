import logging
from rich.logging import RichHandler

from oddkiva.brahma.torch.utils.logging import format_msg


logging.basicConfig(
    level="NOTSET", handlers=[RichHandler()]
)
LOGGER = logging.getLogger(__name__)


def test_log():
    LOGGER.debug(format_msg("Hello, World!"))
    LOGGER.info(format_msg("Hello, World!"))
    LOGGER.warning(format_msg("Hello, World!"))
