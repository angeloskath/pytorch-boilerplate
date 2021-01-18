import torch

from ..factory import ObjectFactory


class SingleOptimizerFactory(ObjectFactory):
    """An optimizer factory for all models that use a single optimizer for all
    the parameters."""
    def add_to_parser(self, parser):
        parser.add_argument(
            "--optimizer",
            choices=["adam", "sgd"],
            default="adam",
            help="Choose the optimizer class (default: adam)"
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="Set the learning rate for the optimizer (default: 0.001)"
        )
        parser.add_argument(
            "--sgd_momentum",
            type=float,
            default=0.9,
            help="Set the momentum for SGD (default: 0.9)"
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0,
            help="Set the strength of the weight decay (default: 0)"
        )

        # TOOD: More optimizers and b1, b2 hyperparameters for Adam

    def from_dict(self, arguments):
        optimizer = arguments["optimizer"]
        model = arguments["experiment"].model

        if optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=arguments["lr"],
                weight_decay=arguments["weight_decay"]
            )
        if optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=arguments["lr"],
                momentum=arguments["sgd_momentum"],
                weight_decay=arguments["weight_decay"]
            )


        raise ValueError("Unsupported optimizer {!r}".format(optimizer))
