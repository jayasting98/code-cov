import argparse

from code_cov import arguments


def main(args: argparse.Namespace) -> None:
    pass


if __name__ == '__main__':
    parser = arguments.create_parser()
    args = parser.parse_args()
    main(args)
