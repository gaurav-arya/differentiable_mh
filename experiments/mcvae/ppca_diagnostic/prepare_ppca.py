"""Prepare a portable PPCA artifact for the Julia Stage-III diagnostic.

This file is deliberately a small bridge around the original Thin et al.
repository.  It does not reimplement the MCVAE package.  It either loads a
compatible IWAE checkpoint or invokes the original training entry point, then
exports the linear model, encoder outputs, and the fixed validation batch in a
format that the Julia diagnostic can read without Python dependencies.

Run from the repository root, for example:

    experiments/mcvae/upstream/venv/bin/python \
        experiments/mcvae/ppca_diagnostic/prepare_ppca.py \
        --checkpoint /path/to/epoch.ckpt

Without ``--checkpoint`` the original nested ``main.py`` is invoked with the
PPCA/IWAE settings used by ``ppca_example.ipynb`` and ``run_scripts/exps.sh``.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[3]
MCVAE_ROOT = ROOT / "experiments" / "mcvae"
THIN_ROOT = Path(
    os.environ.get("THIN_ROOT", str(MCVAE_ROOT / "upstream"))
).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=MCVAE_ROOT / "ppca_diagnostic" / "artifact",
        help="directory receiving the portable artifact",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="compatible IWAE checkpoint; omit to train with the nested source",
    )
    parser.add_argument(
        "--hparams",
        type=Path,
        default=None,
        help="optional checkpoint hparams file, retained as provenance only",
    )
    parser.add_argument(
        "--seed", type=int, default=20260818, help="seed for data binarization and q samples"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="epochs used by the fallback source training"
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=50,
        help="IWAE samples used by the fallback source training",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="GPU count passed to the historical Lightning entry point; defaults to CPU",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="fail instead of invoking the nested training script when no checkpoint is given",
    )
    return parser.parse_args()


def model_kwargs(num_samples: int) -> Dict[str, Any]:
    # This is the exact linear PPCA/IWAE architecture used by the source
    # notebook.  Keeping the constructor explicit also avoids depending on
    # Lightning's historical YAML serialization of Python classes.
    import torch.nn as nn

    return {
        "shape": 28,
        "act_func": nn.GELU,
        "num_samples": num_samples,
        "hidden_dim": 100,
        "net_type": "fc",
        "dataset": "mnist",
        "name": "IWAE",
        "specific_likelihood": "gaussian",
        "sigma": 0.1,
    }


def locate_checkpoint() -> Path:
    candidates = sorted(
        (THIN_ROOT / "lightning_logs").glob("**/checkpoints/*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "nested training completed without producing a checkpoint under "
            f"{THIN_ROOT / 'lightning_logs'}"
        )
    return candidates[-1]


def train_with_nested_source(epochs: int, num_samples: int, seed: int, gpus: int) -> Path:
    command = [
        sys.executable,
        "main.py",
        "--model",
        "IWAE",
        "--dataset",
        "mnist",
        "--binarize",
        "True",
        "--hidden_dim",
        "100",
        "--net_type",
        "fc",
        "--batch_size",
        "100",
        "--val_batch_size",
        "100",
        "--max_epochs",
        str(epochs),
        "--num_samples",
        str(num_samples),
        "--specific_likelihood",
        "gaussian",
        "--sigma",
        "0.1",
        "--gpus",
        str(gpus),
    ]
    environment = os.environ.copy()
    environment["PL_GLOBAL_SEED"] = str(seed)
    # The historical entry point defaults to 20 workers and pinned memory.
    # Keep conservative CPU defaults, but allow a GPU job to override them.
    environment.setdefault("THIN_NUM_WORKERS", "0")
    environment.setdefault("THIN_PIN_MEMORY", "0")
    environment.setdefault("THIN_PERSISTENT_WORKERS", "0")
    print("Running nested source training:", " ".join(command), flush=True)
    subprocess.run(command, cwd=THIN_ROOT, env=environment, check=True)
    return locate_checkpoint()


def checkpoint_state(model: Any, checkpoint_path: Path) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def write_matrix(path: Path, value: Any) -> None:
    import numpy as np

    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    np.savetxt(path, array.reshape(array.shape[0], -1) if array.ndim > 1 else array, delimiter=",")


def write_metadata(path: Path, values: Dict[str, Any]) -> None:
    def toml_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{text}"'

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated by prepare_ppca.py\n")
        for key, value in values.items():
            handle.write(f"{key} = {toml_value(value)}\n")


def prepare(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch

    sys.path.insert(0, str(THIN_ROOT))
    from models.vaes import IWAE
    from utils import make_dataloaders

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint is None:
        if args.no_train:
            raise FileNotFoundError("--no-train was supplied but --checkpoint was not")
        checkpoint = train_with_nested_source(
            args.epochs, args.train_num_samples, args.seed, args.gpus
        )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = IWAE(**model_kwargs(args.train_num_samples))
    checkpoint_state(model, checkpoint)
    model.eval()

    # The nested source resolves its dataset path as ``./data``.  Always make
    # this call from the nested repository, independent of the caller's cwd.
    previous_cwd = Path.cwd()
    os.chdir(THIN_ROOT)
    try:
        _, validation_loader = make_dataloaders(
            dataset="mnist",
            batch_size=100,
            val_batch_size=100,
            binarize=True,
            num_workers=0,
        )
    finally:
        os.chdir(previous_cwd)
    x, labels = next(iter(validation_loader))
    with torch.no_grad():
        mu, logvar = model.encode(x)
        scale = torch.exp(0.5 * logvar)
        z_n1 = mu + scale * torch.randn_like(scale)
        mu_n10 = mu.repeat(10, 1)
        logvar_n10 = logvar.repeat(10, 1)
        z_n10 = mu_n10 + torch.exp(0.5 * logvar_n10) * torch.randn_like(mu_n10)

    decoder = model.decoder_net.net[0]
    encoder = model.encoder_net.net[0]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    write_matrix(output / "decoder_weight.csv", decoder.weight)
    write_matrix(output / "decoder_bias.csv", decoder.bias)
    write_matrix(output / "encoder_weight.csv", encoder.weight)
    write_matrix(output / "encoder_bias.csv", encoder.bias)
    write_matrix(output / "x.csv", x.reshape(x.shape[0], -1))
    write_matrix(output / "labels.csv", labels)
    write_matrix(output / "q_mu.csv", mu)
    write_matrix(output / "q_logvar.csv", logvar)
    write_matrix(output / "z_n1.csv", z_n1)
    write_matrix(output / "z_n10.csv", z_n10)

    metadata = {
        "format_version": 1,
        "latent_dim": 100,
        "observation_dim": 784,
        "batch_size": int(x.shape[0]),
        "sigma": 0.1,
        "source_epsilon": 0.003,
        "julia_mala_gamma": 0.006,
        "training_num_samples": args.train_num_samples,
        "seed": args.seed,
        "dynamically_binarized": True,
        "checkpoint": checkpoint,
        "source_root": THIN_ROOT,
    }
    if args.hparams is not None:
        metadata["hparams"] = args.hparams.resolve()
    write_metadata(output / "metadata.toml", metadata)
    print(f"Wrote PPCA artifact to {output}")
    return output


if __name__ == "__main__":
    prepare(parse_args())
