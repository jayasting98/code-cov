import abc
from typing import Any
from typing import TypedDict

import transformers
from typing_extensions import Required
from typing_extensions import Self

from code_cov import utilities


class ModelInfo(TypedDict, total=False):
    _alias: Required[str]
    config: dict[str, Any]


class ModelFactory(abc.ABC):
    @abc.abstractmethod
    def create(self: Self) -> transformers.PreTrainedModel:
        raise NotImplementedError()


_model_factory_alias_types: dict[str, type[ModelFactory]] = dict()


model_factory = (
    utilities.create_object_alias_decorator(_model_factory_alias_types))


@model_factory('seq_to_seq_lm')
class SeqToSeqLmFactory(ModelFactory):
    def __init__(self: Self, **config: Any) -> None:
        self._config = config

    def create(self: Self) -> transformers.PreTrainedModel:
        model = (
            transformers.AutoModelForSeq2SeqLM.from_pretrained(**self._config))
        return model


def create_model(
    model_info: ModelInfo,
    model_factory_alias_types: dict[str, type[ModelFactory]] = (
        _model_factory_alias_types),
) -> transformers.PreTrainedModel:
    model_factory_alias = model_info['_alias']
    model_factory_type = model_factory_alias_types[model_factory_alias]
    model_factory_config = model_info.get('config', dict())
    model_factory = model_factory_type(**model_factory_config)
    model = model_factory.create()
    return model
