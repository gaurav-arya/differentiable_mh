from argparse import ArgumentParser
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from models import VAE, IWAE, AMCVAE, LMCVAE, VAE_with_flows
from utils import make_dataloaders, get_activations, str2bool

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')

    parser.add_argument("--model", default="VAE",
                        choices=["VAE", "IWAE", "AMCVAE", "LMCVAE", "VAE_with_flows"])

    ## Dataset params
    parser.add_argument("--dataset", default='mnist', choices=['mnist', 'fashionmnist', 'cifar', 'omniglot', 'celeba'])
    parser.add_argument("--binarize", type=str2bool, default=False)
    ## Training parameters
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--val_batch_size", default=50, type=int)
    parser.add_argument("--grad_skip_val", type=float, default=0.)
    parser.add_argument("--grad_clip_val", type=float, default=0.)
    parser.add_argument(
        "--adam_lr",
        type=float,
        default=1e-3,
        help="Initial Adam learning rate; the historical default is 1e-3.",
    )

    ## Architecture
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--act_func", default="gelu",
                        choices=["relu", "leakyrelu", "tanh", "logsigmoid", "logsoftmax", "softplus", "gelu"])
    parser.add_argument("--net_type", choices=["fc", "conv"], type=str, default="conv")

    ## Specific parameters
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--n_leapfrogs", type=int, default=3)
    parser.add_argument("--step_size", type=float, default=0.01)

    parser.add_argument("--use_barker", type=str2bool, default=False)
    parser.add_argument("--use_score_matching", type=str2bool, default=False)  # for ULA
    parser.add_argument("--use_cloned_decoder", type=str2bool,
                        default=False)  # for AIS VAE (to make grad throught alphas easier)
    parser.add_argument("--learnable_transitions", type=str2bool,
                        default=False)  # for AIS VAE and ULA (if learn stepsize or not)
    parser.add_argument("--variance_sensitive_step", type=str2bool,
                        default=False)  # for AIS VAE and ULA (adapt stepsize based on dim's variance)
    parser.add_argument(
        "--bound_variance_sensitive_step",
        type=str2bool,
        default=False,
        help=(
            "bound variance-sensitive per-coordinate step sizes to the "
            "model's declared epsilon_min/epsilon_max; False preserves Thin's baseline"
        ),
    )
    parser.add_argument("--use_alpha_annealing", type=str2bool,
                        default=False)  # for AIS VAE, True if we want anneal sum_log_alphas during training
    parser.add_argument(
        "--gradient_estimator",
        type=str,
        choices=["reinforce", "dmh_one", "dmh_all"],
        default="reinforce",
        help="A-MCVAE accept/reject estimator; reinforce is the historical default",
    )
    parser.add_argument("--annealing_scheme", type=str,
                        default='linear')  # for AIS VAE and ULA VAE, strategy to do annealing
    parser.add_argument("--specific_likelihood", type=str,
                        default=None)  # specific likelihood

    parser.add_argument("--ula_skip_threshold", type=float,
                        default=0.0)  # Probability threshold, if below -- skip transition
    parser.add_argument("--acceptance_rate_target", type=float,
                        default=0.95)  # Target acceptance rate
    parser.add_argument("--sigma", type=float, default=1.)

    parser.add_argument("--num_flows", type=int, default=1)

    act_func = get_activations()

    args = parser.parse_args()
    if args.gradient_estimator != "reinforce" and args.model != "AMCVAE":
        raise ValueError("DMH estimators are currently implemented for AMCVAE only")
    if args.gradient_estimator != "reinforce":
        if args.use_barker:
            raise ValueError("DMH estimators require --use_barker False")
        if args.annealing_scheme != "linear":
            raise ValueError("DMH estimators currently require --annealing_scheme linear")
    print(args)

    def env_bool(name, default):
        value = os.environ.get(name)
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes", "on"}

    num_workers = int(os.environ.get("THIN_NUM_WORKERS", "20"))
    if num_workers < 0:
        raise ValueError("THIN_NUM_WORKERS must be non-negative")
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": env_bool("THIN_PIN_MEMORY", True),
    }
    if num_workers > 0 and env_bool("THIN_PERSISTENT_WORKERS", False):
        kwargs["persistent_workers"] = True
    print("DataLoader settings:", kwargs)
    train_loader, val_loader = make_dataloaders(dataset=args.dataset,
                                                batch_size=args.batch_size,
                                                val_batch_size=args.val_batch_size,
                                                binarize=args.binarize,
                                                **kwargs)
    image_shape = train_loader.dataset.shape_size
    if args.model == "VAE":
        model = VAE(shape=image_shape, act_func=act_func[args.act_func],
                    num_samples=args.num_samples, hidden_dim=args.hidden_dim,
                    net_type=args.net_type, dataset=args.dataset, specific_likelihood=args.specific_likelihood,
                    sigma=args.sigma)
    elif args.model == "IWAE":
        model = IWAE(shape=image_shape, act_func=act_func[args.act_func], num_samples=args.num_samples,
                     hidden_dim=args.hidden_dim,
                     name=args.model, net_type=args.net_type, dataset=args.dataset,
                     specific_likelihood=args.specific_likelihood, sigma=args.sigma)
    elif args.model == "VAE_with_flows":
        model = VAE_with_flows(shape=image_shape, act_func=act_func[args.act_func], num_samples=args.num_samples,
                               hidden_dim=args.hidden_dim, name=args.model, flow_type="RealNVP",
                               num_flows=args.num_flows,
                               net_type=args.net_type, dataset=args.dataset,
                               specific_likelihood=args.specific_likelihood,
                               sigma=args.sigma)
    elif args.model == 'AMCVAE':
        model = AMCVAE(shape=image_shape, step_size=args.step_size, K=args.K, use_barker=args.use_barker,
                       num_samples=args.num_samples, acceptance_rate_target=args.acceptance_rate_target,
                       dataset=args.dataset, net_type=args.net_type, act_func=act_func[args.act_func],
                       hidden_dim=args.hidden_dim, name=args.model, grad_skip_val=args.grad_skip_val,
                       grad_clip_val=args.grad_clip_val,
                       use_cloned_decoder=args.use_cloned_decoder, learnable_transitions=args.learnable_transitions,
                       variance_sensitive_step=args.variance_sensitive_step,
                       bound_variance_sensitive_step=args.bound_variance_sensitive_step,
                       use_alpha_annealing=args.use_alpha_annealing, annealing_scheme=args.annealing_scheme,
                       gradient_estimator=args.gradient_estimator,
                       adam_lr=args.adam_lr,
                       specific_likelihood=args.specific_likelihood, sigma=args.sigma)
    elif args.model == 'LMCVAE':
        model = LMCVAE(shape=image_shape, step_size=args.step_size, K=args.K,
                       num_samples=args.num_samples, acceptance_rate_target=args.acceptance_rate_target,
                       dataset=args.dataset, net_type=args.net_type, act_func=act_func[args.act_func],
                       hidden_dim=args.hidden_dim, name=args.model, grad_skip_val=args.grad_skip_val,
                       grad_clip_val=args.grad_clip_val, use_score_matching=args.use_score_matching,
                       use_cloned_decoder=args.use_cloned_decoder, learnable_transitions=args.learnable_transitions,
                       variance_sensitive_step=args.variance_sensitive_step,
                       bound_variance_sensitive_step=args.bound_variance_sensitive_step,
                       ula_skip_threshold=args.ula_skip_threshold, annealing_scheme=args.annealing_scheme,
                       specific_likelihood=args.specific_likelihood, sigma=args.sigma)
    else:
        raise ValueError

    args.gradient_clip_val = args.grad_clip_val
    automatic_optimization = (
        args.gradient_estimator == "reinforce"
        and args.grad_skip_val == 0.
        and args.gradient_clip_val == 0.
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        fast_dev_run=False,
        terminate_on_nan=automatic_optimization,
        automatic_optimization=automatic_optimization,
    )
    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=val_loader,
    )
