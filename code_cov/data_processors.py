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
    def process(
        self: Self,
        dataset: datasets.Dataset,
        tokenizer: (transformers.PreTrainedTokenizer
            | transformers.PreTrainedTokenizerFast),
    ) -> datasets.Dataset:
        raise NotImplementedError()


_processor_alias_types: dict[str, type[Processor]] = dict()


processor = utilities.create_object_alias_decorator(_processor_alias_types)


@processor('split_non_consecutive')
class NonConsecutiveSplitter(Processor):
    _FOCAL_LINE_KEYS = {'focal_line_indices', 'focal_lines'}

    def process(
        self: Self,
        dataset: datasets.Dataset,
        tokenizer: (transformers.PreTrainedTokenizer
            | transformers.PreTrainedTokenizerFast),
    ) -> datasets.Dataset:
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

    def process(
        self: Self,
        dataset: datasets.Dataset,
        tokenizer: (transformers.PreTrainedTokenizer
            | transformers.PreTrainedTokenizerFast),
    ) -> datasets.Dataset:
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


class DatasetConfig(TypedDict, total=False):
    loader_config: Required[dict[str, Any]]
    processor_infos: list[ProcessorInfo]


def _create_processor(
    processor_info: ProcessorInfo,
    processor_alias_types: dict[str, type[Processor]] = _processor_alias_types,
) -> Processor:
    alias = processor_info['_alias']
    config = processor_info.get('config', dict())
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
        processor = _create_processor(
            processor_info, processor_alias_types=processor_alias_types)
        dataset = processor.process(dataset, tokenizer)
    return dataset
