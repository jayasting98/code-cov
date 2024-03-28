from typing import TypedDict

import transformers


class ModelInfo(TypedDict, total=False):
    pass


def create_model(model_info: ModelInfo) -> transformers.PreTrainedModel:
    pass
