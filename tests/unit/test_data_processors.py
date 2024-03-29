import unittest
from unittest import mock

import datasets
from typing_extensions import Self

import transformers

from code_cov import data_processors


class DataProcessorsTest(unittest.TestCase):
    def test_create_dataset__with_no_processors__creates_correctly(self):
        loader_config = dict(path='path/to/dataset')
        dataset_config = (
            data_processors.DatasetConfig(loader_config=loader_config))
        tokenizer = mock.MagicMock()
        expected_dataset = (
            datasets.Dataset.from_dict(dict(message=['Hello', 'World!'])))
        with mock.patch('datasets.load_dataset') as load_dataset_mock:
            load_dataset_mock.return_value = expected_dataset
            actual_dataset = (
                data_processors.create_dataset(dataset_config, tokenizer))
        load_dataset_mock.assert_called_once_with(path='path/to/dataset')
        actual_iter = iter(actual_dataset)
        self.assertEqual(dict(message='Hello'), next(actual_iter))
        self.assertEqual(dict(message='World!'), next(actual_iter))
        with self.assertRaises(StopIteration):
            next(actual_iter)

    def test_create_dataset__with_processors__creates_correctly(self):
        processor_alias_types = dict()
        @data_processors.processor('stub', alias_objects=processor_alias_types)
        class ProcessorStub(data_processors.Processor):
            def __init__(self, multiplier) -> None:
                self._multiplier = multiplier

            def process(
                self: Self,
                dataset: datasets.Dataset,
                tokenizer: (transformers.PreTrainedTokenizer
                    | transformers.PreTrainedTokenizerFast),
            ) -> datasets.Dataset:
                def multiply(x):
                    x['num'] *= self._multiplier
                    return x
                processed_ds = dataset.map(multiply)
                return processed_ds
        loader_config = dict(path='path/to/dataset')
        processor_infos = [data_processors
            .ProcessorInfo(_alias='stub', config=dict(multiplier=3))]
        dataset_config = data_processors.DatasetConfig(
            loader_config=loader_config, processor_infos=processor_infos)
        tokenizer = mock.MagicMock()
        expected_dataset = (
            datasets.Dataset.from_dict(dict(num=[13, 5, 8])))
        with mock.patch('datasets.load_dataset') as load_dataset_mock:
            load_dataset_mock.return_value = expected_dataset
            actual_dataset = data_processors.create_dataset(dataset_config,
                tokenizer, processor_alias_types=processor_alias_types)
        load_dataset_mock.assert_called_once_with(path='path/to/dataset')
        actual_iter = iter(actual_dataset)
        self.assertEqual(dict(num=39), next(actual_iter))
        self.assertEqual(dict(num=15), next(actual_iter))
        self.assertEqual(dict(num=24), next(actual_iter))
        with self.assertRaises(StopIteration):
            next(actual_iter)
