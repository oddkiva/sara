# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

import pathlib


CONFIG_FILE_PATH= pathlib.Path(__file__).parent / 'config.toml'
if CONFIG_FILE_PATH.exists():
    with open(str(CONFIG_FILE_PATH), 'r') as f:
        from pip._vendor import tomli
        CONFIG = tomli.loads(f.read())
        DATA_DIR_PATH = pathlib.Path(CONFIG['data']['path'])
        assert DATA_DIR_PATH.exists(), \
            "DATA_DIR_PATH from config.toml is invalid"
else:
    THIS_FILE = __file__
    THIS_DIR = pathlib.Path(THIS_FILE).parent
    DATA_DIR_PATH = (pathlib.Path(THIS_DIR) / '../../data').resolve()
    assert DATA_DIR_PATH.exists()
