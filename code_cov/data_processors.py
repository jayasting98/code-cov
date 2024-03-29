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
