from typing import Any
from typing import TypedDict

import datasets
import transformers
from typing_extensions import Required


class DatasetConfig(TypedDict, total=False):
    loader_config: Required[dict[str, Any]]


def create_dataset(
    dataset_config: DatasetConfig,
    tokenizer: (transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast),
) -> datasets.Dataset:
    dataset = datasets.load_dataset(**dataset_config['loader_config'])
    return dataset
