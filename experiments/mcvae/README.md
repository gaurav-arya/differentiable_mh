# MCVAE experiments

This directory contains two related experiments for studying gradients through
Metropolis--Hastings accept/reject decisions in annealed importance sampling:

1. a standalone PPCA diagnostic in `ppca_diagnostic/`; and
2. an end-to-end MNIST MCVAE training experiment in `upstream/`.

Both experiments compare the original REINFORCE with differentiable
Metropolis--Hastings (DMH) estimators.

## PPCA diagnostic

The PPCA diagnostic uses a 784-dimensional MNIST observation model and a
100-dimensional Gaussian latent variable. It estimates a decoder-bias
gradient for finite AIS runs with MALA transitions.
The analytical PPCA gradient is a model-level reference.

The Julia code uses the repository's `experiments` project. This release does
not include a generated PPCA artifact or result files.

To run the numerical diagnostic, first prepare an artifact from a compatible
Thin/IWAE checkpoint using the upstreamcode. The Python bridge writes the
portable CSV/TOML bundle consumed by the Julia code:

```bash
python experiments/mcvae/ppca_diagnostic/prepare_ppca.py \
  --checkpoint /path/to/epoch.ckpt \
  --output /path/to/ppca-artifact

julia --project=experiments \
  experiments/mcvae/ppca_diagnostic/run_ppca.jl \
  --artifact=/path/to/ppca-artifact
```

The result is `ppca_results.csv`, `ppca_samples.csv`, and `ppca_validation.csv`;
pass the summary CSV to `plot_ppca_csv.jl` to render a figure.

## End-to-end MNIST experiment

`upstream/main.py` is the training entry point for the adapted MCVAE source.
It supports the model families `VAE`, `IWAE`, `AMCVAE`, `LMCVAE`, and
`VAE_with_flows`. DMH is implemented for `AMCVAE` through the
`--gradient_estimator` option, with choices `reinforce`, `dmh_one`, and
`dmh_all`.

The production comparison uses the following six configurations. The DMH runs
use one latent particle; the REINFORCE baselines use five.

| Configuration | Estimator | AIS steps (`K`) | Particles |
|---|---|---:|---:|
| `amcvae3_linear` | REINFORCE | 3 | 5 |
| `amcvae5_linear` | REINFORCE | 5 | 5 |
| `amcvae3_dmh_one` | DMH-one | 3 | 1 |
| `amcvae3_dmh_all` | DMH-all | 3 | 1 |
| `amcvae5_dmh_one` | DMH-one | 5 | 1 |
| `amcvae5_dmh_all` | DMH-all | 5 | 1 |

The command below is the equivalent direct training command for
`amcvae3_dmh_all`. For `dmh_one`, change `--gradient_estimator` accordingly;
for `K=5`, change `--K 3` to `--K 5`. The REINFORCE baselines use
`--gradient_estimator reinforce` and `--num_samples 5`.

Run from the upstream directory so that the dataset and Lightning output are
placed there:

```bash
cd experiments/mcvae/upstream
python main.py \
  --model AMCVAE \
  --dataset mnist \
  --binarize True \
  --hidden_dim 64 \
  --batch_size 100 \
  --val_batch_size 50 \
  --net_type conv \
  --num_samples 1 \
  --max_epochs 100 \
  --step_size 0.01 \
  --K 3 \
  --variance_sensitive_step True \
  --bound_variance_sensitive_step True \
  --use_barker False \
  --acceptance_rate_target 0.8 \
  --use_alpha_annealing True \
  --annealing_scheme linear \
  --gradient_estimator dmh_all \
  --grad_clip_val 50 \
  --adam_lr 1e-3 \
  --gpus 1
```

MNIST is obtained through the dataset loader in `upstream/utils`;
the required Python dependencies, including the PyTorch and PyTorch
Lightning versions used by the adapted source, are not bundled in this
release.

## Attribution

The `upstream/` source is a reduced, adapted subset of the MCVAE implementation
from the [original MCVAE repository](https://github.com/stat-ml/mcvae),
accompanying:

> Achille Thin, Nikita Kotelevskii, Arnaud Doucet, Alain Durmus, Eric Moulines,
> and Maxim Panov. “Monte Carlo Variational Auto-Encoders.” *Proceedings of
> the 38th International Conference on Machine Learning*, PMLR 139,
> 10247–10257, 2021.

Paper: [PMLR](https://proceedings.mlr.press/v139/thin21a.html)

```bibtex
@inproceedings{thin2021monte,
  title = {Monte Carlo Variational Auto-Encoders},
  author = {Thin, Achille and Kotelevskii, Nikita and Doucet, Arnaud and
            Durmus, Alain and Moulines, Eric and Panov, Maxim},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages = {10247--10257},
  year = {2021},
  publisher = {PMLR},
  volume = {139}
}
```

The DMH citations can be found in the root README. 
Please cite Thin et al. when using the MCVAE source or PPCA setup, and cite the
DMH papers when using the differentiable Metropolis--Hastings estimators.
