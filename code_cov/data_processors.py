from typing import TypedDict

import datasets
import transformers


class DatasetConfig(TypedDict, total=False):
    pass


def create_dataset(
    dataset_config: DatasetConfig,
    tokenizer: (transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast),
) -> datasets.Dataset:
    pass
