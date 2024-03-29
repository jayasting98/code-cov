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
