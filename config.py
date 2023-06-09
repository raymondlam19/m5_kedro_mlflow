import json
import os
from configparser import ConfigParser
from pathlib import Path


class CustomConfigParser(ConfigParser):
    def __init__(self):
        super(ConfigParser, self).__init__()

    def getlist(self, section: str, key: str) -> list:
        return self.get(section, key).split(",")

    def get_dict(self, section, key):
        return json.loads(self.get(section, key).replace("'", '"'))

    def get_path(self, section, key):
        return Path(self.get(section, key))


HOME_PATH = os.environ.get("HOME_PATH", "")
config = CustomConfigParser()
config.read(Path(HOME_PATH) / "env_config.ini")

# To use them
# from config import HOME_PATH, config

if __name__ == "__main__":
    if config.get("env_param", "testing"):
        print(HOME_PATH)
    else:
        print("testing=False in env_config.ini")
