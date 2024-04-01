import unittest
from unittest import mock

import transformers
from typing_extensions import Self

from code_cov import data_collator_factories


class DataCollatorFactoriesTest(unittest.TestCase):
    def test_create_data_collator__typical_case__creates_correctly(self):
        data_collator_factory_alias_types = dict()
        data_collator_mock = mock.MagicMock()
        @data_collator_factories.data_collator_factory(
            'stub', alias_objects=data_collator_factory_alias_types)
        class DataCollatorFactoryStub(
            data_collator_factories.DataCollatorFactory):
            def __init__(
                self: Self,
                tokenizer: (transformers.PreTrainedTokenizer
                    | transformers.PreTrainedTokenizerFast),
                model: transformers.PreTrainedModel,
            ) -> None:
                pass

            def create(self: Self) -> data_collator_factories.DataCollator:
                return data_collator_mock
        data_collator_info = (
            data_collator_factories.DataCollatorInfo(_alias='stub'))
        tokenizer_mock = mock.MagicMock()
        model_mock = mock.MagicMock()
        actual_data_collator = data_collator_factories.create_data_collator(
            data_collator_info, tokenizer_mock, model_mock,
            data_collator_factory_alias_types=data_collator_factory_alias_types)
        self.assertIs(data_collator_mock, actual_data_collator)


class SeqToSeqDataCollatorFactoryTest(unittest.TestCase):
    def test_create_data_collator__typical_case__creates_correctly(self):
        data_collator_factory_config = dict(return_tensors='pt')
        data_collator_info = data_collator_factories.DataCollatorInfo(
            _alias='seq_to_seq', config=data_collator_factory_config)
        tokenizer_mock = mock.MagicMock()
        model_mock = mock.MagicMock()
        expected_data_collator = mock.MagicMock()
        with mock.patch('transformers.DataCollatorForSeq2Seq') as init_mock:
            init_mock.return_value = expected_data_collator
            actual_data_collator = data_collator_factories.create_data_collator(
                data_collator_info, tokenizer_mock, model_mock)
        init_mock.assert_called_once_with(
            tokenizer_mock, model=model_mock, return_tensors='pt')
        self.assertIs(expected_data_collator, actual_data_collator)
