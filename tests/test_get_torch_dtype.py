import pytest
import torch

import torchutils

@pytest.mark.parametrize("precision", ["fp16", "bf16", "no"])
def test_get_torch_dtype(precision: str) -> None:
    if (
        (precision == "bf16")
        and not (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
    ):
        pytest.skip("bf16 precision was requested but not available on this hardware.")
    torch_dtypes = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }
    expected = torch_dtypes[precision]

    assert torchutils.get_torch_dtype(precision) == expected
