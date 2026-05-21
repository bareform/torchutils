import os
import sys

import pytest
import tomli_w

import torchutils

@pytest.mark.parametrize("dataset", ["CIFAR10"])
@pytest.mark.parametrize("num_epoch", [100])
@pytest.mark.parametrize("in_channels", [[3, 64, 128, 256]])
@pytest.mark.parametrize("fc", [[256]])
@pytest.mark.parametrize("dropout", [0.5])
@pytest.mark.parametrize("metrics", [["accuracy"]])
@pytest.mark.parametrize("checkpoint", ["./checkpoints"])
def test_config_parser(
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    num_epoch: int, 
    in_channels: list[int],
    fc: list[int],
    dropout: float,
    metrics: list[str],
    checkpoint: str,
) -> None:
    config_data = {
        "dataset": "CIFAR10",
        "training": {
            "num_epoch": 100,
        },
        "model": {
            "in_channels": [3, 64, 128, 256],
            "fc": [256],
            "dropout": 0.5,
        },
        "metrics": ["accuracy"],
        "checkpoint": "./checkpoints",
    }
    config_dir = "configs"
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "config.toml")
    with open(config_file, "wb") as fp:
        tomli_w.dump(config_data, fp)

    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--config", config_file,
    ])

    parser = torchutils.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10"],
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["accuracy"],
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        nargs="+",
        default=[3, 32, 64, 128],
    )
    parser.add_argument(
        "--fc",
        type=int,
        nargs="+",
        default=[128],
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()
    
    assert args.dataset == "CIFAR10"
    assert args.num_epoch == 100
    assert args.in_channels == [3, 64, 128, 256]
    assert args.fc == [256]
    assert args.dropout == 0.5
    assert args.metrics == ["accuracy"]
    assert args.checkpoint == "./checkpoints"
