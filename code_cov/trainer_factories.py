from typing import TypedDict

import datasets
import transformers


class TrainerInfo(TypedDict, total=False):
    pass


def create_trainer(
    trainer_info: TrainerInfo,
    tokenizer: (transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast),
    dataset: datasets.Dataset,
    training_arguments: transformers.TrainingArguments,
    model: transformers.PreTrainedModel,
) -> transformers.Trainer:
    pass
