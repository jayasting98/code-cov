import unittest
from unittest import mock

from code_cov import trainer_factories


class TrainerFactoriesTest(unittest.TestCase):
    def test_create_trainer__typical_case__creates_correctly(self):
        data_collator_config = dict(return_tensors='pt')
        trainer_info = dict(data_collator_config=data_collator_config)
        expected_tokenizer = mock.MagicMock()
        expected_dataset = mock.MagicMock()
        expected_training_arguments = mock.MagicMock()
        expected_model = mock.MagicMock()
        expected_data_collator = mock.MagicMock()
        expected_trainer = mock.MagicMock()
        with (mock.patch('transformers.DataCollatorForSeq2Seq')
                as data_collator_init_mock,
            mock.patch('transformers.Trainer') as trainer_init_mock):
            data_collator_init_mock.return_value = expected_data_collator
            trainer_init_mock.return_value = expected_trainer
            actual_trainer = trainer_factories.create_trainer(trainer_info,
                expected_tokenizer, expected_dataset,
                expected_training_arguments, expected_model)
        data_collator_init_mock.assert_called_once_with(
            expected_tokenizer, model=expected_model, return_tensors='pt')
        trainer_init_mock.assert_called_once_with(model=expected_model,
            args=expected_training_arguments,
            data_collator=expected_data_collator,
            train_dataset=expected_dataset, tokenizer=expected_tokenizer)
        self.assertIs(expected_trainer, actual_trainer)
