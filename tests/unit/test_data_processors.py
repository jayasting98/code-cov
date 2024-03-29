import unittest
from unittest import mock

import datasets

from code_cov import data_processors


class DataProcessorsTest(unittest.TestCase):
    def test_create_dataset__typical_case__creates_correctly(self):
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
