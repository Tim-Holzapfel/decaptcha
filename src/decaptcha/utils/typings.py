# Standard Library
from typing import Any, Mapping, TypedDict

# Thirdparty Library
from torch.nn.modules import loss


class Checkpoint(TypedDict):
    model_state_dict: Mapping[str, Any]
    optimizer_state_dict: dict[str, Any]
    loss: loss.CrossEntropyLoss
    epoch: int
