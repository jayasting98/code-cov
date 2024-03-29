import abc
from typing import Any
from typing import TypedDict

import datasets
import transformers
from typing_extensions import Required
from typing_extensions import Self

from code_cov import utilities


class ProcessorInfo(TypedDict, total=False):
    _alias: Required[str]
    config: dict[str, Any]


class Processor(abc.ABC):
    @abc.abstractmethod
    def process(self: Self, dataset: datasets.Dataset) -> datasets.Dataset:
        raise NotImplementedError()


_processor_alias_types: dict[str, type[Processor]] = dict()


processor = utilities.create_object_alias_decorator(_processor_alias_types)


@processor('split_non_consecutive')
class NonConsecutiveSplitter(Processor):
    _FOCAL_LINE_KEYS = {'focal_line_indices', 'focal_lines'}

    def process(self: Self, dataset: datasets.Dataset) -> datasets.Dataset:
        processed_ds = dataset.map(self._split_non_consecutive, batched=True)
        return processed_ds

    def _split_non_consecutive(
        self: Self,
        samples: dict[str, list[Any]],
    ) -> dict[str, list[Any]]:
        split_samples = {key: [] for key in samples.keys()}
        num_samples = len(samples['focal_line_indices'])
        for i in range(num_samples):
            focal_line_indices = samples['focal_line_indices'][i]
            focal_lines = samples['focal_lines'][i]
            consecutive_line_indices = []
            consecutive_lines = []
            num_focal_lines = len(focal_lines)
            m = num_focal_lines - 1
            for j in range(m):
                focal_line_index = focal_line_indices[j]
                consecutive_line_indices.append(focal_line_index)
                consecutive_lines.append(focal_lines[j])
                if focal_line_index + 1 == focal_line_indices[j + 1]:
                    continue
                for key in samples.keys():
                    if key in self._FOCAL_LINE_KEYS:
                        continue
                    split_samples[key].append(samples[key][i])
                (split_samples['focal_line_indices']
                    .append(consecutive_line_indices))
                split_samples['focal_lines'].append(consecutive_lines)
                consecutive_line_indices = []
                consecutive_lines = []
            consecutive_line_indices.append(focal_line_indices[m])
            consecutive_lines.append(focal_lines[m])
            for key in samples.keys():
                if key in self._FOCAL_LINE_KEYS:
                    continue
                split_samples[key].append(samples[key][i])
            split_samples['focal_line_indices'].append(consecutive_line_indices)
            split_samples['focal_lines'].append(consecutive_lines)
        return split_samples


@processor('generate_prompt_and_target')
class PromptAndTargetGenerator(Processor):
    _OPENING_TAG_COMMENT_TEMPLATE: str = '/* <{name}> */'
    _CLOSING_TAG_COMMENT_TEMPLATE: str = '/* </{name}> */'

    def __init__(
        self: Self,
        prompt_template: str,
        focal_method_tag_name: str,
        focal_lines_tag_name: str,
        test_input_method_tag_name: str,
        test_target_method_tag_name: str,
    ) -> None:
        self._prompt_template = prompt_template
        self._focal_method_tag_name = focal_method_tag_name
        self._focal_lines_tag_name = focal_lines_tag_name
        self._test_input_method_tag_name = test_input_method_tag_name
        self._test_target_method_tag_name = test_target_method_tag_name

    def process(self: Self, dataset: datasets.Dataset) -> datasets.Dataset:
        processed_ds = dataset.map(
            self._generate_prompt_and_target,
            remove_columns=dataset.column_names)
        return processed_ds

    def _generate_prompt_and_target(
        self: Self,
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        prompt_focal_class = sample['focal_class']['identifier']
        focal_method = sample['focal_method']
        focal_method_line_start = focal_method['line_start']
        focal_method_body: str = focal_method['body']
        focal_line_indices = sample['focal_line_indices']
        indices = [focal_line_index - focal_method_line_start
            for focal_line_index in focal_line_indices]
        focal_method_lines = focal_method_body.split('\n')
        focal_method_closing_tag_comment = (
            self._create_closing_tag_comment(self._focal_method_tag_name))
        focal_method_lines.insert(
            len(focal_method_lines), focal_method_closing_tag_comment)
        focal_lines_closing_tag_comment = (
            self._create_closing_tag_comment(self._focal_lines_tag_name))
        focal_method_lines.insert(
            indices[-1] + 1, focal_lines_closing_tag_comment)
        focal_lines_opening_tag_comment = (
            self._create_opening_tag_comment(self._focal_lines_tag_name))
        focal_method_lines.insert(
            indices[0], focal_lines_opening_tag_comment)
        focal_method_opening_tag_comment = (
            self._create_opening_tag_comment(self._focal_method_tag_name))
        focal_method_lines.insert(0, focal_method_opening_tag_comment)
        test_input_method = sample['test_input_method']
        test_input_method_body: str = test_input_method['body']
        test_input_method_lines = test_input_method_body.split('\n')
        test_input_method_closing_tag_comment = (
            self._create_closing_tag_comment(self._test_input_method_tag_name))
        test_input_method_lines.insert(
            len(test_input_method_lines), test_input_method_closing_tag_comment)
        test_input_method_opening_tag_comment = (
            self._create_opening_tag_comment(self._test_input_method_tag_name))
        test_input_method_lines.insert(
            0, test_input_method_opening_tag_comment)
        test_target_method = sample['test_target_method']
        test_target_method_body: str = test_target_method['body']
        test_target_method_lines = test_target_method_body.split('\n')
        test_target_method_closing_tag_comment = (
            self._create_closing_tag_comment(self._test_target_method_tag_name))
        test_target_method_lines.insert(len(test_target_method_lines),
            test_target_method_closing_tag_comment)
        test_target_method_opening_tag_comment = (
            self._create_opening_tag_comment(self._test_target_method_tag_name))
        test_target_method_lines.insert(
            0, test_target_method_opening_tag_comment)
        prompt_focal_method = '\n'.join(focal_method_lines)
        prompt_test_input_method = '\n'.join(test_input_method_lines)
        prompt_test_target_method = '\n'.join(test_target_method_lines)
        prompt = self._prompt_template.format(focal_class=prompt_focal_class,
            focal_method=prompt_focal_method,
            test_input_method=prompt_test_input_method)
        prompt_sample = dict(prompt=prompt, target=prompt_test_target_method)
        return prompt_sample

    def _create_opening_tag_comment(self: Self, name: str) -> str:
        opening_tag_comment = (
            self._create_tag_comment(self._OPENING_TAG_COMMENT_TEMPLATE, name))
        return opening_tag_comment

    def _create_closing_tag_comment(self: Self, name: str) -> str:
        closing_tag_comment = (
            self._create_tag_comment(self._CLOSING_TAG_COMMENT_TEMPLATE, name))
        return closing_tag_comment

    def _create_tag_comment(self: Self, template: str, name: str) -> str:
        tag_comment = template.format(name=name)
        return tag_comment


@processor('tokenize_for_seq2seq')
class Seq2SeqTokenizer(Processor):
    def __init__(
        self: Self,
        input_key: str,
        target_key: str,
        tokenizer: (transformers.PreTrainedTokenizer
            | transformers.PreTrainedTokenizerFast),
        tokenizer_config: dict[str, Any],
        label_pad_token_id: int = -100,
    ) -> None:
        self._input_key = input_key
        self._target_key = target_key
        self._tokenizer = tokenizer
        self._tokenizer_config = tokenizer_config
        self._label_pad_token_id = label_pad_token_id

    def process(self: Self, dataset: datasets.Dataset) -> datasets.Dataset:
        processed_ds = dataset.map(
            self._tokenize, batched=True, remove_columns=dataset.column_names)
        return processed_ds

    def _tokenize(self: Self, samples: dict[str, list[Any]]):
        inputs = samples[self._input_key]
        targets = samples[self._target_key]
        tokenized_samples = self._tokenizer(
            text=inputs, text_target=targets, **self._tokenizer_config)
        formatted_labels = []
        for token_ids in tokenized_samples['labels']:
            formatted_token_ids = []
            for token_id in token_ids:
                formatted_token_id: int
                if token_id == self._tokenizer.pad_token_id:
                    formatted_token_id = self._label_pad_token_id
                else:
                    formatted_token_id = token_id
                formatted_token_ids.append(formatted_token_id)
            formatted_labels.append(formatted_token_ids)
        tokenized_samples['labels'] = formatted_labels
        return tokenized_samples


class DatasetConfig(TypedDict, total=False):
    loader_config: Required[dict[str, Any]]
    processor_infos: list[ProcessorInfo]


def _create_processor(
    processor_info: ProcessorInfo,
    tokenizer: (transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast),
    processor_alias_types: dict[str, type[Processor]] = _processor_alias_types,
) -> Processor:
    alias = processor_info['_alias']
    config = processor_info.get('config', dict())
    if 'tokenizer_config' in config:
        config['tokenizer'] = tokenizer
    type_ = processor_alias_types[alias]
    processor = type_(**config)
    return processor


def create_dataset(
    dataset_config: DatasetConfig,
    tokenizer: (transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast),
    processor_alias_types: dict[str, type[Processor]] = _processor_alias_types,
) -> datasets.Dataset:
    dataset = datasets.load_dataset(**dataset_config['loader_config'])
    processor_infos = dataset_config.get('processor_infos', list())
    for processor_info in processor_infos:
        processor = _create_processor(processor_info,
            tokenizer, processor_alias_types=processor_alias_types)
        dataset = processor.process(dataset)
    return dataset
