import os
import yaml
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

class Settings:
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent


@dataclass
class DataConfig:
    """Configuration class for data loading and processing parameters.

    Attributes:
        path (Dict[str, str]): Dictionary containing data directory paths.
            Contains base_dir, raw, interim, and feature paths.
        name (Dict[str, str]): Dictionary containing dataset names.
            Currently includes bgsf2 dataset name.
        loading (Dict[str, str]): Dictionary containing data loading parameters.
            Includes file_extension and encoding settings.
    """
    path: Dict[str, str]
    # name: Dict[str, str] 
    # loading: Dict[str, str]

    @classmethod
    def from_yaml(cls, config_path: str = "config/data.yaml") -> 'DataConfig':
        """Load configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.
                Defaults to "data_config.yaml".

        Returns:
            DataConfig: Configuration object with data parameters.
        """
        with open(f"{Settings.PROJECT_ROOT}/{config_path}", 'r') as f:
            config = yaml.safe_load(f)
        for attr in cls.__annotations__:
            setattr(cls, attr, config[attr])
        cls.path = {k: f"{Settings.PROJECT_ROOT}/{v}" for k, v in cls.path.items()}
        return cls


print(DataConfig.from_yaml().path)
