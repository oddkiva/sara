import pathlib


CONFIG_FILE_PATH= pathlib.Path(__file__).parent / 'config.toml'
if CONFIG_FILE_PATH.exists():
    with open(CONFIG_FILE_PATH, 'rb') as f:
        from pip._vendor import tomli
        CONFIG = tomli.load(f)
        DATA_DIR_PATH = pathlib.Path(CONFIG['data']['path'])
