import pytest
import torch

import torchutils

@pytest.mark.parametrize("precision", ["fp16", "bf16", "no"])
def test_get_torch_dtype(precision: str) -> None:
    expected = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }[precision]

    assert torchutils.get_torch_dtype(precision) == expected
