from typing import Any
from typing import TypedDict

import datasets
import transformers


class TrainerInfo(TypedDict, total=False):
    data_collator_config: dict[str, Any]


def create_trainer(
    trainer_info: TrainerInfo,
    tokenizer: (transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast),
    dataset: datasets.Dataset,
    training_arguments: transformers.TrainingArguments,
    model: transformers.PreTrainedModel,
) -> transformers.Trainer:
    data_collator_config = trainer_info.get('data_collator_config', dict())
    data_collator = (transformers
        .DataCollatorForSeq2Seq(tokenizer, model=model, **data_collator_config))
    trainer = transformers.Trainer(model=model, args=training_arguments,
        data_collator=data_collator, train_dataset=dataset, tokenizer=tokenizer)
    return trainer
