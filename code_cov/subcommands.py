import argparse
import json
import logging
import math
import os
import pathlib
import re
import tempfile
import traceback
from typing import Any

import git
import torch
import transformers
from typing_extensions import Self

from code_cov import arguments
from code_cov import coverages
from code_cov import data_collator_factories
from code_cov import data_processors
from code_cov import model_factories
from code_cov import projects
from code_cov import utilities


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


@arguments.subcommand('evaluate')
class EvaluateSubcommand(arguments.Subcommand):
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
        dataset_config = (
            data_processors.DatasetConfig(**config['dataset_config']))
        try:
            dataset_config['loader_config']['storage_options']['token'] = (
                self._token)
        except KeyError:
            # If the key 'storage_options' does not exist, then the token does
            # not apply, so we can just pass.
            pass
        code_cov_config = config['code_cov_config']
        output_file_pathname = config['output_path']
        skip = config['skip']
        limit = config['limit']
        regexps = config['regexps']
        test_candidate_method_re = re.compile(regexps['candidate'])
        test_method_name_re = re.compile(regexps['test_method_name'])
        test_class_body_re = re.compile(regexps['test_class_body'])
        logging.info('creating dataset')
        dataset = data_processors.create_dataset(dataset_config, None)
        logging.info('creating code cov')
        code_cov = coverages.CodeCovCli(**code_cov_config)
        ds_iter = iter(dataset)
        for _ in zip(range(skip), ds_iter):
            pass
        repository_url_datas = dict()
        for i, sample in zip(range(skip, skip + limit), ds_iter):
            repository = sample['repository']
            url = repository['repository_url']
            if url not in repository_url_datas:
                logging.info(f'sample {i}: repository {url}')
                temp_dir = tempfile.TemporaryDirectory()
                project_path_datas = dict()
                repository_data = dict(dir=temp_dir,
                    projects=project_path_datas)
                repository_url_datas[url] = repository_data
                try:
                    logging.info(f'sample {i}: cloning')
                    git.Repo.clone_from(url, temp_dir.name)
                except Exception as exception:
                    logging.warning(f'{exception}: {url}')
                    logging.debug(f'{traceback.format_exc()}')
                    continue
            repository_data = repository_url_datas[url]
            project_path_datas = repository_data['projects']
            temp_dir = repository_data['dir']
            project_relative_pathname = sample['project_path']
            project_pathname = (
                os.path.join(temp_dir.name, project_relative_pathname))
            if project_relative_pathname not in project_path_datas:
                logging.info(f'sample {i}: project {project_relative_pathname}')
                project = (
                    projects.create_project(temp_dir.name, project_pathname))
                logging.info(f'sample {i}: finding classpath pathnames')
                classpath_pathnames = project.find_classpath_pathnames()
                logging.info(f'sample {i}: finding focal classpath')
                focal_classpath = project.find_focal_classpath()
                project_data = dict(project=project,
                    classpath_pathnames=classpath_pathnames,
                    focal_classpath=focal_classpath,
                    sample_statistics=dict())
                project_path_datas[project_relative_pathname] = project_data
            project_data = project_path_datas[project_relative_pathname]
            project: projects.Project = project_data['project']
            classpath_pathnames = project_data['classpath_pathnames']
            focal_classpath = project_data['focal_classpath']
            focal_class = sample['focal_class']
            focal_package = focal_class['package']
            focal_class_identifier = focal_class['identifier']
            focal_class_name = f'{focal_package}.{focal_class_identifier}'
            test_class = sample['test_class']
            test_package = test_class['package']
            test_class_identifier = test_class['identifier']
            test_class_name = f'{test_package}.{test_class_identifier}'
            test_file_relative_pathname = sample['test_file']
            test_file_pathname = (
                os.path.join(project_pathname, test_file_relative_pathname))
            candidate = sample['candidate']
            focal_line_indices = sample['focal_line_indices']
            sample_id = (sample['focal_method']['identifier'],
                sample['test_input_method']['identifier'],
                sample['test_target_method']['identifier'],
                focal_line_indices[0], focal_line_indices[-1])
            if sample_id not in project_data['sample_statistics']:
                logging.info(f'sample {i}: {sample_id}')
                project_data['sample_statistics'][sample_id] = (
                    dict(total=0, correct=0))
            project_data['sample_statistics'][sample_id]['total'] += 1
            test_candidate_method_match = (
                test_candidate_method_re.search(candidate))
            if test_candidate_method_match is None:
                # incorrect sample
                logging.info(f'sample {i}: test candidate method not found')
                logging.debug(f'{candidate}')
                continue
            test_candidate_method = test_candidate_method_match[1]
            logging.info(f'sample {i}: test candidate method found')
            logging.debug(f'{test_candidate_method}')
            test_candidate_method_name_match = (
                test_method_name_re.search(test_candidate_method))
            if test_candidate_method_name_match is None:
                # incorrect sample
                logging.info(
                    f'sample {i}: test candidate method name not found')
                continue
            test_candidate_method_name = test_candidate_method_name_match[1]
            logging.info(f'sample {i}: test candidate method name found '
                + f'`{test_candidate_method_name}`')
            with utilities.TemporaryChangeFile(test_file_pathname):
                with open(test_file_pathname) as test_file:
                    test_file_content = test_file.read()
                test_class_body_match = (
                    test_class_body_re.search(test_file_content))
                if test_class_body_match is None:
                    logging.warning(f'sample {i}: test class not found: '
                        + f'{test_file_pathname}')
                    continue
                test_class_body = test_class_body_match[1]
                test_candidate_file_content = (test_file_content
                    .replace(test_class_body, test_candidate_method))
                with open(test_file_pathname, mode='w') as test_file:
                    test_file.write(test_candidate_file_content)
                logging.info(f'sample {i}: compiling')
                try:
                    project.compile()
                except Exception:
                    # incorrect sample
                    logging.info(f'sample {i}: failed to compile')
                    continue
                request_data = coverages.CreateCoverageRequestData(
                    classpathPathnames=classpath_pathnames,
                    focalClasspath=focal_classpath,
                    focalClassName=focal_class_name,
                    testClassName=test_class_name,
                    testMethodName=test_candidate_method_name,
                )
                try:
                    with utilities.WorkingDirectory(project_pathname):
                        coverage = code_cov.create_coverage(request_data)
                except Exception as exception:
                    # incorrect sample
                    logging.info(f'sample {i}: coverage not found')
                    continue
            is_fully_covered = True
            for focal_line_index in focal_line_indices:
                focal_line_number = focal_line_index + 1
                if focal_line_number not in coverage['coveredLineNumbers']:
                    logging.info(
                        f'sample {i}: line {focal_line_number} not covered')
                    is_fully_covered = False
                    break
            if not is_fully_covered:
                # incorrect sample
                continue
            logging.info(f'sample {i}: correct')
            project_data['sample_statistics'][sample_id]['correct'] += 1
        output_data: dict[str, Any] = dict()
        for repository_url, repository_data in repository_url_datas.items():
            temp_dir: tempfile.TemporaryDirectory = repository_data['dir']
            temp_dir.cleanup()
            project_path_datas: dict[str, Any] = repository_data['projects']
            output_repository_data: dict[str, list] = dict()
            for project_id, project_data in project_path_datas.items():
                output_project_data = list()
                sample_statistics: dict[str, dict[str, int]] = (
                    project_data['sample_statistics'])
                for sample_id, sample_data in sample_statistics.items():
                    output_sample_data = dict(**sample_data)
                    output_sample_data['focal_method'] = sample_id[0]
                    output_sample_data['test_input_method'] = sample_id[1]
                    output_sample_data['test_target_method'] = sample_id[2]
                    output_sample_data['line_index_start'] = sample_id[3]
                    output_sample_data['line_index_end'] = sample_id[4]
                    output_project_data.append(output_sample_data)
                output_repository_data[project_id] = output_project_data
            output_data[repository_url] = output_repository_data
        (pathlib.Path(output_file_pathname)
            .parent.mkdir(parents=True, exist_ok=True))
        with open(output_file_pathname, mode='w') as output_file:
            json.dump(output_data, output_file, indent=4)
