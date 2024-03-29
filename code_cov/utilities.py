from collections.abc import Callable
from typing import Any
from typing import Generic
from typing import Protocol
from typing import TypeVar

from typing_extensions import Self


_T = TypeVar('_T')


class _ObjectAliasDecorator(Protocol, Generic[_T]):
    def __call__(
        self: Self,
        alias: str,
        alias_objects: dict[str, _T] = None,
    ) -> Callable[[_T], _T]:
        pass


def create_object_alias_decorator(
    _alias_objects: dict[str, _T],
) -> _ObjectAliasDecorator[_T]:
    def object_alias(
        alias: str,
        alias_objects: dict[str, _T] = None,
    ) -> Callable[[_T], _T]:
        if alias_objects is None:
            alias_objects = _alias_objects
        def decorator(obj: _T) -> _T:
            alias_objects[alias] = obj
            return obj
        return decorator
    return object_alias
