"""
Configuration registry for training configs.
"""

from pydantic import BaseModel

# Global registry of training config classes
_TRAINING_CONFIG_REGISTRY: dict[str, type[BaseModel]] = {}


def register_training_config(name: str):
    """Decorator to register a training config class in the registry.

    Usage:
        @register_training_config("MyTrainingConfig")
        class MyTrainingConfig(SPOCTrainingConfig):
            ...

    Args:
        name: The name to register this config under (used in command line)
    """

    def decorator(cls: type[BaseModel]):
        if name in _TRAINING_CONFIG_REGISTRY:
            print(f"Warning: Overriding existing training config '{name}'")
        _TRAINING_CONFIG_REGISTRY[name] = cls
        return cls

    return decorator


def get_training_config_class(name: str) -> type[BaseModel]:
    """Get a training config class by name from the registry."""
    if name not in _TRAINING_CONFIG_REGISTRY:
        available = list(_TRAINING_CONFIG_REGISTRY.keys())
        raise ValueError(
            f"Training config '{name}' not found. Available configs: {available}"
        )
    return _TRAINING_CONFIG_REGISTRY[name]


def list_available_training_configs() -> list[str]:
    """List all available training config names in the registry."""
    return list(_TRAINING_CONFIG_REGISTRY.keys())


def get_training_registry_size() -> int:
    """Get the number of registered training configs."""
    return len(_TRAINING_CONFIG_REGISTRY)
