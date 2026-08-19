

from dataclasses import fields, is_dataclass
from typing import Any

from torch.utils import _pytree

_REGISTERED_PYTREE_DATACLASSES: set[type[Any]] = set()


def register_pytree_dataclass_type(cls: type[Any]) -> None:
    """Register a dataclass type so torch pytree can recurse into it."""
    if cls in _REGISTERED_PYTREE_DATACLASSES:
        return
    if not is_dataclass(cls):
        return

    field_names = tuple(field.name for field in fields(cls))

    def _flatten(instance: Any):
        k_v_pairs = [(name, getattr(instance, name)) for name in field_names]
        values = [v for _, v in k_v_pairs if v is not None]
        keys = [k for k, v in k_v_pairs if v is not None]
        return values, keys

    def _unflatten(values: list[Any], context: tuple[str, ...]):
        kwargs = dict(zip(context, values))
        return cls(**kwargs)

    try:
        _pytree.register_pytree_node(cls, _flatten, _unflatten)
    except ValueError:
        # Already registered in this process.
        pass
    _REGISTERED_PYTREE_DATACLASSES.add(cls)


def register_pytree_dataclasses(obj: Any) -> None:
    """Recursively register dataclass instances contained in an object tree."""
    if is_dataclass(obj) and not isinstance(obj, type):
        register_pytree_dataclass_type(type(obj))
        for field in fields(obj):
            register_pytree_dataclasses(getattr(obj, field.name))
        return
    if isinstance(obj, dict):
        for value in obj.values():
            register_pytree_dataclasses(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            register_pytree_dataclasses(value)
