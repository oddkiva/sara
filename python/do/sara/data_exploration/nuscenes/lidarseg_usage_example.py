from os import path
from pathlib import Path

from nuscenes import NuScenes

NUSCENES_ROOT_PATH = path.join(str(Path.home()), 'Downloads', 'nuscenes')

nuscenes = NuScenes(version='v1.0-mini',
                    dataroot=NUSCENES_ROOT_PATH,
                    verbose=True)

