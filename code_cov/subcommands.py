import argparse
import json
import logging
import math
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


@arguments.subcommand('generate')
class GenerateSubcommand(arguments.Subcommand):
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
            # If the key 'storage_options' does not exist, then the token does
            # not apply, so we can just pass.
            pass
        model_info = model_factories.ModelInfo(**config['model_info'])
        generation_config: dict[str, Any] = config['generation_config']
        num_candidates: int = generation_config['num_candidates']
        generation_batch_size: int = generation_config['batch_size']
        generate_config: dict[str, Any] = generation_config['generate_config']
        map_config = config['map_config']
        saver_config = config['saver_config']
        try:
            saver_config['storage_options']['token'] = self._token
        except KeyError:
            # If the key 'storage_options' does not exist, then the token does
            # not apply, so we can just pass.
            pass
        logging.info('creating tokenizer')
        tokenizer = (
            transformers.AutoTokenizer.from_pretrained(**tokenizer_config))
        model_max_length = tokenizer.model_max_length
        logging.info(f'model max length: {model_max_length}')
        logging.info('creating dataset')
        dataset = data_processors.create_dataset(dataset_config, tokenizer)
        logging.info('creating devices')
        cpu = torch.device('cpu')
        device = torch.device('cuda') if torch.cuda.is_available() else cpu
        logging.info(
            f'cuda max memory allocated: {torch.cuda.max_memory_allocated()}')
        logging.info('creating model')
        model = model_factories.create_model(model_info).to(device)
        logging.info(
            f'cuda max memory allocated: {torch.cuda.max_memory_allocated()}')
        has_max_new_tokens = 'max_new_tokens' in generate_config
        max_new_tokens_: int
        if has_max_new_tokens:
            max_new_tokens_ = generate_config.pop('max_new_tokens')
        def generate(samples: dict[str, list[Any]]) -> dict[str, list[Any]]:
            candidate_samples = {key: list() for key in samples.keys()}
            candidate_samples['generated_ids'] = list()
            candidate_samples['candidate'] = list()
            inputs = samples['input_ids']
            batch_size = len(inputs)
            for i in range(batch_size):
                logging.info(f'i: {i}')
                input_ids = torch.LongTensor([inputs[i]]).to(device)
                if has_max_new_tokens:
                    max_new_tokens = max_new_tokens_
                else:
                    labels = samples['labels'][i]
                    num_target_tokens = (
                        sum([1 for label in labels if label != 0]))
                    logging.info(f'num_target_tokens: {num_target_tokens}')
                    # Round up to the second nearest larger power of two
                    # for allowance.
                    power = math.log(num_target_tokens) / math.log(2)
                    rounded_power = math.ceil(power) + 1
                    max_new_tokens = 2 ** rounded_power
                    # It should not exceed the model max length.
                    max_new_tokens = min(max_new_tokens, model_max_length)
                logging.info(f'max_new_tokens: {max_new_tokens}')
                for j in range(0, num_candidates, generation_batch_size):
                    logging.info(f'j: {j}')
                    num_return_sequences = (
                        min(generation_batch_size, num_candidates - j))
                    outputs = model.generate(input_ids=input_ids,
                        num_return_sequences=num_return_sequences,
                        max_new_tokens=max_new_tokens,
                        **generate_config)
                    logging.info('cuda max memory allocated: '
                        + f'{torch.cuda.max_memory_allocated()}')
                    for generated_ids in outputs:
                        for key in samples.keys():
                            candidate_samples[key].append(samples[key][i])
                        candidate = (tokenizer
                            .decode(generated_ids, skip_special_tokens=True))
                        sample_generated_ids = generated_ids.to(cpu).tolist()
                        (candidate_samples['generated_ids']
                            .append(sample_generated_ids))
                        candidate_samples['candidate'].append(candidate)
            logging.info(
                f'candidate_samples len: {len(candidate_samples["candidate"])}')
            logging.info('cuda max memory allocated: '
                + f'{torch.cuda.max_memory_allocated()}')
            return candidate_samples
        candidate_ds = dataset.map(generate, batched=True, **map_config)
        candidate_ds.save_to_disk(**saver_config)
