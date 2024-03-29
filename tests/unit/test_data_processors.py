import unittest
from unittest import mock

import datasets
from typing_extensions import Self

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


class NonConsecutiveSplitterTest(unittest.TestCase):
    def test_create_dataset__typical_case__creates_correctly(self):
        loader_config = dict(path='path/to/dataset')
        processor_infos = [
            data_processors.ProcessorInfo(_alias='split_non_consecutive')]
        dataset_config = data_processors.DatasetConfig(
            loader_config=loader_config, processor_infos=processor_infos)
        tokenizer = mock.MagicMock()
        expected_dataset = datasets.Dataset.from_dict(dict(
            focal_line_indices=[
                [
                    5, 6, 7,
                    40, 41, 42, 43,
                    527, 528,
                ],
                [
                    3,
                    5,
                    8,
                    13,
                ],
            ],
            focal_lines=[
                [
                    'some', 'focal', 'lines',
                    'these', 'are', 'consecutive', 'ones',
                    'Hello', 'World!',
                ],
                [
                    'these',
                    'are',
                    'not',
                    'consecutive',
                ],
            ],
            other_key=[
                'this is some value',
                'another one here',
            ],
        ))
        with mock.patch('datasets.load_dataset') as load_dataset_mock:
            load_dataset_mock.return_value = expected_dataset
            actual_dataset = (
                data_processors.create_dataset(dataset_config, tokenizer))
        actual_iter = iter(actual_dataset)
        expected_sample_0 = dict(
            focal_line_indices=[5, 6, 7],
            focal_lines=['some', 'focal', 'lines'],
            other_key='this is some value',
        )
        self.assertEqual(expected_sample_0, next(actual_iter))
        expected_sample_1 = dict(
            focal_line_indices=[40, 41, 42, 43],
            focal_lines=['these', 'are', 'consecutive', 'ones'],
            other_key='this is some value',
        )
        self.assertEqual(expected_sample_1, next(actual_iter))
        expected_sample_2 = dict(
            focal_line_indices=[527, 528],
            focal_lines=['Hello', 'World!'],
            other_key='this is some value',
        )
        self.assertEqual(expected_sample_2, next(actual_iter))
        expected_sample_3 = dict(
            focal_line_indices=[3],
            focal_lines=['these'],
            other_key='another one here',
        )
        self.assertEqual(expected_sample_3, next(actual_iter))
        expected_sample_4 = dict(
            focal_line_indices=[5],
            focal_lines=['are'],
            other_key='another one here',
        )
        self.assertEqual(expected_sample_4, next(actual_iter))
        expected_sample_5 = dict(
            focal_line_indices=[8],
            focal_lines=['not'],
            other_key='another one here',
        )
        self.assertEqual(expected_sample_5, next(actual_iter))
        expected_sample_6 = dict(
            focal_line_indices=[13],
            focal_lines=['consecutive'],
            other_key='another one here',
        )
        self.assertEqual(expected_sample_6, next(actual_iter))
        with self.assertRaises(StopIteration):
            next(actual_iter)


class PromptAndTargetGeneratorTest(unittest.TestCase):
    def test_create_dataset__typical_case__creates_correctly(self):
        loader_config = dict(path='path/to/dataset')
        processor_config = dict(
            prompt_template=(''
                + '// Focal class: {focal_class}\n'
                + '// Focal method:\n'
                + '{focal_method}\n'
                + '// Test method:\n'
                + '{test_input_method}'),
            focal_method_tag_name='method',
            focal_lines_tag_name='code',
            test_input_method_tag_name='input',
            test_target_method_tag_name='edited',
        )
        processor_info = data_processors.ProcessorInfo(
            _alias='generate_prompt_and_target', config=processor_config)
        processor_infos = [processor_info]
        dataset_config = data_processors.DatasetConfig(
            loader_config=loader_config, processor_infos=processor_infos)
        tokenizer = mock.MagicMock()
        expected_dataset = datasets.Dataset.from_dict(dict(
            focal_class=[
                dict(identifier='SomeClass'),
                dict(identifier='SomeOtherClass'),
            ],
            focal_method=[
                dict(
                    line_start=13,
                    body=('static void isEven(int x) {\n'
                        + '        if (x % 2 == 0) {\n'
                        + '            boolean value = true;\n'
                        + '            return value;\n'
                        + '        } else {\n'
                        + '            return false;\n'
                        + '        }\n'
                        + '    }'),
                ),
                dict(
                    line_start=34,
                    body=('static void isOdd(int x) {\n'
                        + '        if (x % 2 == 0) {\n'
                        + '            boolean value = false;\n'
                        + '            return value;\n'
                        + '        } else {\n'
                        + '            return true;\n'
                        + '        }\n'
                        + '    }'),
                ),
            ],
            focal_line_indices=[
                [15, 16],
                [39],
            ],
            test_input_method=[
                dict(
                    body=('@Test\n'
                        + '    void testIsEven_odd_false() {\n'
                        + '        assertFalse(SomeClass.isEven(5));\n'
                        + '    }'),
                ),
                dict(
                    body=('@Test\n'
                        + '    void testIsOdd_even_false() {\n'
                        + '        assertFalse(SomeOtherClass.isOdd(8));\n'
                        + '    }'),
                ),
            ],
            test_target_method=[
                dict(
                    body=('@Test\n'
                        + '    void testIsEven_even_true() {\n'
                        + '        assertTrue(SomeClass.isEven(8));\n'
                        + '    }'),
                ),
                dict(
                    body=('@Test\n'
                        + '    void testIsOdd_odd_true() {\n'
                        + '        assertTrue(SomeOtherClass.isOdd(5));\n'
                        + '    }'),
                ),
            ],
            other_key=[
                'is',
                'ignored',
            ]
        ))
        with mock.patch('datasets.load_dataset') as load_dataset_mock:
            load_dataset_mock.return_value = expected_dataset
            actual_dataset = (
                data_processors.create_dataset(dataset_config, tokenizer))
        actual_iter = iter(actual_dataset)
        expected_sample_0 = dict(
            prompt=(''
                + '// Focal class: SomeClass\n'
                + '// Focal method:\n'
                + '/* <method> */\n'
                + 'static void isEven(int x) {\n'
                + '        if (x % 2 == 0) {\n'
                + '/* <code> */\n'
                + '            boolean value = true;\n'
                + '            return value;\n'
                + '/* </code> */\n'
                + '        } else {\n'
                + '            return false;\n'
                + '        }\n'
                + '    }\n'
                + '/* </method> */\n'
                + '// Test method:\n'
                + '/* <input> */\n'
                + '@Test\n'
                + '    void testIsEven_odd_false() {\n'
                + '        assertFalse(SomeClass.isEven(5));\n'
                + '    }\n'
                + '/* </input> */'),
            target=(''
                + '/* <edited> */\n'
                + '@Test\n'
                + '    void testIsEven_even_true() {\n'
                + '        assertTrue(SomeClass.isEven(8));\n'
                + '    }\n'
                + '/* </edited> */'),
        )
        self.assertEqual(expected_sample_0, next(actual_iter))
        expected_sample_1 = dict(
            prompt=(''
                + '// Focal class: SomeOtherClass\n'
                + '// Focal method:\n'
                + '/* <method> */\n'
                + 'static void isOdd(int x) {\n'
                + '        if (x % 2 == 0) {\n'
                + '            boolean value = false;\n'
                + '            return value;\n'
                + '        } else {\n'
                + '/* <code> */\n'
                + '            return true;\n'
                + '/* </code> */\n'
                + '        }\n'
                + '    }\n'
                + '/* </method> */\n'
                + '// Test method:\n'
                + '/* <input> */\n'
                + '@Test\n'
                + '    void testIsOdd_even_false() {\n'
                + '        assertFalse(SomeOtherClass.isOdd(8));\n'
                + '    }\n'
                + '/* </input> */'),
            target=(''
                + '/* <edited> */\n'
                + '@Test\n'
                + '    void testIsOdd_odd_true() {\n'
                + '        assertTrue(SomeOtherClass.isOdd(5));\n'
                + '    }\n'
                + '/* </edited> */'),
        )
        self.assertEqual(expected_sample_1, next(actual_iter))
        with self.assertRaises(StopIteration):
            next(actual_iter)


class Seq2SeqTokenizerTest(unittest.TestCase):
    def test_create_dataset__typical_case__creates_correctly(self):
        loader_config = dict(path='path/to/dataset')
        processor_config = dict(
            input_key='prompt',
            target_key='output',
            tokenizer_config=dict(
                padding='max_length',
            ),
        )
        processor_info = data_processors.ProcessorInfo(
            _alias='tokenize_for_seq2seq', config=processor_config)
        processor_infos = [processor_info]
        dataset_config = data_processors.DatasetConfig(
            loader_config=loader_config, processor_infos=processor_infos)
        tokenizer = mock.MagicMock()
        tokenizer.return_value = dict(
            input_ids=[
                [7, 8, 9, 10, 11, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 8, 9, 10, 11, 12, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            labels=[
                [3, 5, 13, 8, 34, 21, 144, 55, 89, 0, 0, 0, 0, 0, 0, 0],
                [3, 5, 13, 8, 55, 21, 144, 34, 89, 0, 0, 0, 0, 0, 0, 0],
            ],
        )
        tokenizer.pad_token_id = 0
        expected_dataset = datasets.Dataset.from_dict(dict(
            prompt=[
                'this is a prompt',
                'another input is here',
            ],
            output=[
                'you call this an output',
                'the one here as well',
            ],
        ))
        with mock.patch('datasets.load_dataset') as load_dataset_mock:
            load_dataset_mock.return_value = expected_dataset
            actual_dataset = (
                data_processors.create_dataset(dataset_config, tokenizer))
        tokenizer.assert_called_once_with(
            text=[
                'this is a prompt',
                'another input is here',
            ],
            text_target=[
                'you call this an output',
                'the one here as well',
            ],
            padding='max_length',
        )
        actual_iter = iter(actual_dataset)
        expected_sample_0 = dict(
            input_ids=[7, 8, 9, 10, 11, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            labels=[3, 5, 13, 8, 34, 21, 144, 55, 89, -100, -100, -100, -100,
                -100, -100, -100],
        )
        self.assertEqual(expected_sample_0, next(actual_iter))
        expected_sample_1 = dict(
            input_ids=[5, 8, 9, 10, 11, 12, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            labels=[3, 5, 13, 8, 55, 21, 144, 34, 89, -100, -100, -100, -100,
                -100, -100, -100],
        )
        self.assertEqual(expected_sample_1, next(actual_iter))
        with self.assertRaises(StopIteration):
            next(actual_iter)
