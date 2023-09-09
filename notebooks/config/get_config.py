import yaml
import os
from configparser import ConfigParser
from pathlib import Path

# Set env variables on Windows or in launch.json and settings.json
HOME_PATH = os.environ.get("HOME_PATH", "")


class ConfigHandler:
    """
    Read env_config.ini / .yml
    """

    @classmethod
    def get(cls, section: str, key: str):
        """
        Get the value of the key under the section from env_config.ini

        section : section in env_config.ini (e.g. "env_param")
        key     : key under the section (e.g. "model_params_path")
        """
        config = ConfigParser()
        config.read(Path(HOME_PATH) / "env_config.ini")
        return config.get(section, key)

    @classmethod
    def read_yml(cls, key_or_path: str):
        """
        Read yml

        key_or_path: either a key in env_config.ini or a relative path of the yml
        """
        try:
            config_path = HOME_PATH / Path(cls.get("env_param", key_or_path))
        except:
            config_path = HOME_PATH / Path(key_or_path)
        with open(config_path, "r") as f:
            yml_file = yaml.load(f, yaml.FullLoader)
        return yml_file


if __name__ == "__main__":
    # Unit test
    print(f"HOME_PATH: {HOME_PATH}")

    if ConfigHandler.get("env_param", "testing"):
        print("pass get")

    if ConfigHandler.read_yml("model_params_path"):
        model_params = ConfigHandler.read_yml("model_params_path")
        print("pass yml", model_params["common"])

    if ConfigHandler.read_yml("notebooks/config/model_params.yml"):
        model_params = ConfigHandler.read_yml("notebooks/config/model_params.yml")
        print("pass yml", model_params["lgbm"]["lgbm_params"])
