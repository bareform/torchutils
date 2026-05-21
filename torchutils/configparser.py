import argparse
from typing import Any, Sequence

import tomli as tomllib

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        args = super().parse_args(args, namespace)
        def flatten(data: dict[str, Any]) -> dict[str, Any]:
            items: dict[str, Any] = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    items.update(flatten(value))
                else:
                    items[key] = value
            return items

        if hasattr(args, "config") and args.config:
            with open(args.config, "rb") as f:
                config_values = tomllib.load(f)
            config = flatten(config_values)
            for k, v in config.items():
                if hasattr(args, k):
                    current = getattr(args, k)
                    if current == self.get_default(k):
                        setattr(args, k, v)
        return args
