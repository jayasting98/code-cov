import argparse
import json
import logging
from typing import Any

import torch
import transformers
from typing_extensions import Self

from code_cov import arguments
from code_cov import data_collator_factories
from code_cov import data_processors
from code_cov import model_factories


@arguments.subcommand('train')
class TrainSubcommand(arguments.Subcommand):
    def __init__(self: Self, args: argparse.Namespace) -> None:
        self._config_file_pathname: str = args.config_path
        self._log_level: str = args.log_level.upper()
        self._token: dict[str, str] = args.token

    @classmethod
    def setup_parser(cls: type[Self], parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--config_path', required=True)
        parser.add_argument('--log_level', default='warning')
        parser.add_argument('--token', default=None, type=json.loads)

    def run(self: Self) -> None:
        logging.basicConfig(
            format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
            level=self._log_level,
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        logging.info('parsing config')
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
        data_collator_info = (data_collator_factories
            .DataCollatorInfo(**config['data_collator_info']))
        logging.info('creating tokenizer')
        tokenizer = (
            transformers.AutoTokenizer.from_pretrained(**tokenizer_config))
        logging.info(f'model max length: {tokenizer.model_max_length}')
        logging.info('creating dataset')
        dataset = data_processors.create_dataset(dataset_config, tokenizer)
        logging.info('creating training arguments')
        training_arguments = (
            transformers.TrainingArguments(**training_arguments_config))
        logging.info(f'training_arguments_config: {training_arguments_config}')
        logging.info('creating model')
        model = model_factories.create_model(model_info)
        logging.info('creating data collator')
        data_collator = (data_collator_factories
            .create_data_collator(data_collator_info, tokenizer, model))
        logging.info('creating trainer')
        trainer = transformers.Trainer(model=model, args=training_arguments,
            data_collator=data_collator, train_dataset=dataset,
            tokenizer=tokenizer)
        logging.info(
            f'cuda max memory allocated: {torch.cuda.max_memory_allocated()}')
        logging.info('training')
        trainer.train()
        logging.info(
            f'cuda max memory allocated: {torch.cuda.max_memory_allocated()}')
