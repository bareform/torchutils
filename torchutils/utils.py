
import random

import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_torch_dtype(precision: str | torch.dtype) -> torch.dtype:
    if (
        (precision == "bf16" or precision == torch.bfloat16)
        and not (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
    ):
        raise RuntimeError("bf16 precision was requested but not available on this hardware. Please use fp32 or fp16 precision instead.")
    if isinstance(precision, torch.dtype):
        return precision
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32
