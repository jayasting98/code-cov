import argparse
import json
from typing import Any

import transformers
from typing_extensions import Self

from code_cov import arguments
from code_cov import data_processors
from code_cov import model_factories
from code_cov import trainer_factories


@arguments.subcommand('train')
class TrainSubcommand(arguments.Subcommand):
    def __init__(self: Self, args: argparse.Namespace) -> None:
        self._config_file_pathname: str = args.config_path
        self._token: dict[str, str] = args.token

    @classmethod
    def setup_parser(cls: type[Self], parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--config_path', required=True)
        parser.add_argument('--token', default=None, type=json.loads)

    def run(self: Self) -> None:
        with open(self._config_file_pathname) as config_file:
            config = json.load(config_file)
        tokenizer_config: dict[str, Any] = config['tokenizer_config']
        dataset_config = (
            data_processors.DatasetConfig(**config['dataset_config']))
        try:
            dataset_config['loader_config']['storage_options']['token'] = (
                self._token)
        except KeyError:
            pass
        training_arguments_config: dict[str, Any] = (
            config['training_arguments_config'])
        model_info = model_factories.ModelInfo(**config['model_info'])
        trainer_info = (
            trainer_factories.TrainerInfo(**config['trainer_info']))
        tokenizer = (
            transformers.AutoTokenizer.from_pretrained(**tokenizer_config))
        dataset = data_processors.create_dataset(dataset_config, tokenizer)
        training_arguments = (
            transformers.TrainingArguments(**training_arguments_config))
        model = model_factories.create_model(model_info)
        trainer = trainer_factories.create_trainer(
            trainer_info, tokenizer, dataset, training_arguments, model)
        trainer.train()
