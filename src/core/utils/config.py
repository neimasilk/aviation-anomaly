"""
Configuration loader with environment variable support.
Loads from YAML config files and .env for secrets.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager for the project."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to the YAML config file. If None, uses default.
        """
        # Load environment variables from .env
        load_dotenv()

        # Determine project root
        self.project_root = Path(__file__).parent.parent.parent

        # Load YAML config
        if config_path is None:
            config_path = self.project_root / "config" / "default.yaml"
        else:
            config_path = Path(config_path)

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-separated key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def get_env(self, key: str, default: Any = None) -> str:
        """Get environment variable."""
        return os.getenv(key, default)

    @property
    def deepseek_api_key(self) -> str:
        """Get DeepSeek API key from environment."""
        key = self.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found in environment. "
                "Please set it in .env file."
            )
        return key

    @property
    def deepseek_base_url(self) -> str:
        """Get DeepSeek base URL."""
        return self.get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    @property
    def device(self) -> str:
        """Get device for training."""
        device = self.get_env("DEVICE", "cpu")
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return device

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.project_root / self.get("paths.data_dir", "data")

    @property
    def models_dir(self) -> Path:
        """Get models directory path."""
        return self.project_root / self.get("paths.models_dir", "models")

    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        return self.project_root / self.get("paths.logs_dir", "logs")

    def to_dict(self) -> Dict[str, Any]:
        """Return full config as dictionary."""
        return self._config.copy()


# Global config instance
config = Config()


if __name__ == "__main__":
    # Test config loading
    print("Project root:", config.project_root)
    print("Data dir:", config.data_dir)
    print("Device:", config.device)
    print("\nConfig dump:")
    import json
    print(json.dumps(config.to_dict(), indent=2))
