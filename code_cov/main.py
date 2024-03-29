import argparse

import dotenv

from code_cov import arguments


def main(args: argparse.Namespace) -> None:
    dotenv.load_dotenv(override=True)
    subcommand = arguments.create_subcommand(args.subcommand, args)
    subcommand.run()


if __name__ == '__main__':
    parser = arguments.create_parser()
    args = parser.parse_args()
    main(args)
