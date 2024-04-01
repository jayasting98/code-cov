import abc
from typing import Any
from typing import TypedDict

import datasets
import transformers
from typing_extensions import Required
from typing_extensions import Self

from code_cov import utilities


DataCollator = (transformers.DefaultDataCollator
    | transformers.DataCollatorWithPadding
    | transformers.DataCollatorForTokenClassification
    | transformers.DataCollatorForSeq2Seq
    | transformers.DataCollatorForLanguageModeling
    | transformers.DataCollatorForWholeWordMask
    | transformers.DataCollatorForPermutationLanguageModeling)


class DataCollatorInfo(TypedDict, total=False):
    _alias: Required[str]
    config: dict[str, Any]


class DataCollatorFactory(abc.ABC):
    @abc.abstractmethod
    def create(self: Self) -> DataCollator:
        raise NotImplementedError()


_data_collator_factory_alias_types: dict[str, type[DataCollatorFactory]] = (
    dict())


data_collator_factory = (
    utilities.create_object_alias_decorator(_data_collator_factory_alias_types))


@data_collator_factory('seq_to_seq')
class SeqToSeqDataCollatorFactory(DataCollatorFactory):
    def __init__(
        self: Self,
        tokenizer: (transformers.PreTrainedTokenizer
            | transformers.PreTrainedTokenizerFast),
        model: transformers.PreTrainedModel,
        **config: Any,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._config = config

    def create(self: Self) -> DataCollator:
        data_collator = transformers.DataCollatorForSeq2Seq(
            self._tokenizer, model=self._model, **self._config)
        return data_collator


def create_data_collator(
    data_collator_info: DataCollatorInfo,
    tokenizer: (transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast),
    model: transformers.PreTrainedModel,
    data_collator_factory_alias_types: dict[str, type[DataCollatorFactory]] = (
        _data_collator_factory_alias_types),
) -> DataCollator:
    data_collator_alias = data_collator_info['_alias']
    data_collator_factory_type = (
        data_collator_factory_alias_types[data_collator_alias])
    data_collator_factory_config = data_collator_info.get('config', dict())
    data_collator_factory_config['tokenizer'] = tokenizer
    data_collator_factory_config['model'] = model
    data_collator_factory = (
        data_collator_factory_type(**data_collator_factory_config))
    data_collator = data_collator_factory.create()
    return data_collator
