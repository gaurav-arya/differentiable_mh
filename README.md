![](plots/gaussian_plot_alts_bare.png)

[![arXiv article](https://img.shields.io/badge/article-arXiv%3A10.48550-B31B1B)](https://arxiv.org/abs/2306.07961)

# Differentiable Metropolis-Hastings 

This repository contains the code to reproduce the experiments in our working paper [Differentiating Metropolis-Hastings to Optimize Intractable Densities](https://arxiv.org/abs/2306.07961).

## Abstract

When performing inference on probabilistic models, target densities often become intractable, necessitating the use of Monte Carlo samplers. We develop a methodology for unbiased differentiation of the Metropolis-Hastings sampler, allowing us to differentiate through probabilistic inference. By fusing recent advances in stochastic differentiation with Markov chain coupling schemes, the procedure can be made unbiased, low-variance, and automatic. This allows us to apply gradient-based optimization to objectives expressed as expectations over intractable target densities. We demonstrate our approach by finding an ambiguous observation in a Gaussian mixture model and by maximizing the specific heat in an Ising model.

## Citation

```
@inproceedings{arya2023differentiating,
    title={Differentiating Metropolis-Hastings to Optimize Intractable Densities},
    author={Gaurav Arya and Ruben Seyer and Frank Sch{\"a}fer and Kartik Chandra and Alexander K. Lew and Mathieu Huot and Vikash Mansinghka and Jonathan Ragan-Kelley and Christopher Vincent Rackauckas and Moritz Schauer},
    booktitle={ICML 2023 Workshop on Differentiable Almost Everything: Differentiable Relaxations, Algorithms, Operators, and Simulators},
    year={2023},
    url={https://openreview.net/forum?id=2jag4Yatsz}
}
```

## Reproducing plots

To reproduce the plots in the `plots` folder:

* Enter the `scripts` folder.
* Run `julia _setup_env.jl` to setup your environment.
* Run `julia {script name}.jl` for each of the four scripts to produce the plots.
