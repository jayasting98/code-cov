import unittest
from unittest import mock

import transformers
from typing_extensions import Self

from code_cov import model_factories


class ModelFactoriesTest(unittest.TestCase):
    def test_create_model__typical_case__creates_correctly(self):
        model_factory_alias_types = dict()
        model_mock = mock.MagicMock()
        @model_factories.model_factory(
            'stub', alias_objects=model_factory_alias_types)
        class ModelFactoryStub(model_factories.ModelFactory):
            def create(self: Self) -> transformers.PreTrainedModel:
                return model_mock
        model_info = model_factories.ModelInfo(_alias='stub')
        actual_model = model_factories.create_model(
            model_info, model_factory_alias_types=model_factory_alias_types)
        self.assertIs(model_mock, actual_model)


class SeqToSeqLmFactoryTest(unittest.TestCase):
    def test_create_model__typical_case__creates_correctly(self):
        model_factory_config = dict(
            pretrained_model_name_or_path='Salesforce/codet5p-220m',
            trust_remote_code=True,
        )
        model_info = (model_factories
            .ModelInfo(_alias='seq_to_seq_lm', config=model_factory_config))
        expected_model = mock.MagicMock()
        with (mock.patch('transformers.AutoModelForSeq2SeqLM.from_pretrained')
            as from_pretrained_mock):
            from_pretrained_mock.return_value = expected_model
            actual_model = model_factories.create_model(model_info)
        from_pretrained_mock.assert_called_once_with(
            pretrained_model_name_or_path='Salesforce/codet5p-220m',
            trust_remote_code=True,
        )
        self.assertIs(expected_model, actual_model)
