import argparse
import unittest

from typing_extensions import Self

from code_cov import arguments


class ArgumentsTest(unittest.TestCase):
    def test_create_parser__typical_case__creates_successfully(self):
        subcommand_name_types = dict()
        self.assertEqual(dict(), subcommand_name_types)
        @arguments.subcommand(
            'command', alias_objects=subcommand_name_types)
        class SubcommandStub(arguments.Subcommand):
            @classmethod
            def setup_parser(
                cls: type[Self],
                parser: argparse.ArgumentParser,
            ) -> None:
                pass
        parser = (arguments
            .create_parser(subcommand_name_types=subcommand_name_types))
        args = parser.parse_args(['command'])
        self.assertEqual('command', args.subcommand)

    def test_create_subcommand__typical_case__creates_successfully(self):
        subcommand_name_types = dict()
        self.assertEqual(dict(), subcommand_name_types)
        @arguments.subcommand(
            'command', alias_objects=subcommand_name_types)
        class SubcommandStub(arguments.Subcommand):
            def __init__(self: Self, args: argparse.Namespace) -> None:
                pass

            @classmethod
            def setup_parser(
                cls: type[Self],
                parser: argparse.ArgumentParser,
            ) -> None:
                pass

            def run(self: Self) -> None:
                pass
        parser = (arguments
            .create_parser(subcommand_name_types=subcommand_name_types))
        args = parser.parse_args(['command'])
        subcommand = arguments.create_subcommand(
            args.subcommand, args, subcommand_name_types=subcommand_name_types)
        self.assertIsInstance(subcommand, SubcommandStub)
