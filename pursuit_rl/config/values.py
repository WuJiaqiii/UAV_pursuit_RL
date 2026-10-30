"""Recursively immutable configuration values and detached dictionary access."""

import copy
from collections.abc import Mapping
from types import MappingProxyType


def as_dict(config):
    return config.to_dict() if isinstance(config, ConfigNode) else config


class ConfigNode(Mapping):
    """Attribute and mapping access with recursively read-only values."""

    __slots__ = ("_values",)

    def __init__(self, values):
        def freeze(value):
            if isinstance(value, dict):
                return ConfigNode(value)
            if isinstance(value, list):
                return tuple(freeze(item) for item in value)
            return copy.deepcopy(value)

        object.__setattr__(
            self,
            "_values",
            MappingProxyType({key: freeze(value) for key, value in values.items()}),
        )

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __getattr__(self, key):
        try:
            return self._values[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key, value):
        raise AttributeError(
            "Config is read-only; use with_overrides() to create a validated copy"
        )

    def __delattr__(self, key):
        raise AttributeError("Config is read-only")

    def to_dict(self):
        def thaw(value):
            if isinstance(value, ConfigNode):
                return value.to_dict()
            if isinstance(value, tuple):
                return [thaw(item) for item in value]
            return copy.deepcopy(value)

        return {key: thaw(value) for key, value in self._values.items()}
