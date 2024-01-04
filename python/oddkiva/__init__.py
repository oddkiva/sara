import pathlib
import tomllib


CONFIG_FILE_PATH= pathlib.Path(__file__).parent / 'config.toml'
with open(CONFIG_FILE_PATH, 'rb') as f:
    CONFIG = tomllib.load(f)
    DATA_DIR_PATH = pathlib.Path(CONFIG['data']['path'])
