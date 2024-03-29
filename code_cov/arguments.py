import abc
import argparse

from typing_extensions import Self

from code_cov import utilities


class Subcommand(abc.ABC):
    @abc.abstractmethod
    def __init__(self: Self, args: argparse.Namespace) -> None:
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def setup_parser(cls: type[Self], parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def run(self: Self) -> None:
        raise NotImplementedError()


_subcommand_name_types: dict[str, type[Subcommand]] = dict()


subcommand = utilities.create_object_alias_decorator(_subcommand_name_types)


def create_parser(
    subcommand_name_types: dict[str, type[Subcommand]] = _subcommand_name_types,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand')
    for subcommand_name, subcommand_type in subcommand_name_types.items():
        subparser = subparsers.add_parser(subcommand_name)
        subcommand_type = subcommand_type.setup_parser(subparser)
    return parser


def create_subcommand(
    name: str,
    args: argparse.Namespace,
    subcommand_name_types: dict[str, type[Subcommand]] = _subcommand_name_types,
) -> Subcommand:
    subcommand_type = subcommand_name_types[name]
    subcommand = subcommand_type(args)
    return subcommand
