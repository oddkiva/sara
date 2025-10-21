from typing import Tuple, overload

import numpy as np


@overload
class VideoStream:

    def __init__(self): ...

    def open(self, video_filepath: str, autorotate: bool = True): ...

    def sizes(self) -> Tuple[int, int, int]: ...

    def read(self, video_frame: np.ndarray) -> None: ...
