from collections.abc import Callable
import os
from types import TracebackType
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


class WorkingDirectory:
    """A context manager for executing code in a specified working directory."""
    def __init__(self: Self, working_dir_pathname: str) -> None:
        self._working_dir_pathname = working_dir_pathname

    def __enter__(self: Self) -> None:
        self._original_working_dir_pathname = os.getcwd()
        os.chdir(self._working_dir_pathname)

    def __exit__(
        self: Self,
        exception_type: type[BaseException],
        exception_value: BaseException,
        exception_traceback: TracebackType,
    ) -> bool:
        os.chdir(self._original_working_dir_pathname)
        return False
