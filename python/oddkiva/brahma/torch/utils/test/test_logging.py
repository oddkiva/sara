import logging
from rich.logging import RichHandler

from oddkiva.brahma.torch.utils.logging import logd, logi, logw


logging.basicConfig(
    level="NOTSET", handlers=[RichHandler()]
)
LOGGER = logging.getLogger(__name__)


def test_log():
    logd(LOGGER, "Hello, World!")
    logi(LOGGER, "Hello, World!")
    logw(LOGGER, "Hello, World!")
