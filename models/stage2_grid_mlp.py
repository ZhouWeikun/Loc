import torch
import torch.nn as nn

from models.cond_modulator_shallow_serial import SerialModulatorShallow
from models.multi_mlp import create_mlp, init_weights


class PlainGridMLP(nn.Module):
    """Plain MLP for stage2 grid features, with optional coord feature concat."""

    def __init__(
            self,
            input_dim,
            condition_dim=0,
            hidden_dim=512,
            num_blocks=1,
            output_dim=None,
            use_coord_condition=True,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = int(input_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.output_dim = int(output_dim)
        self.use_coord_condition = bool(use_coord_condition)

        total_input_dim = self.input_dim + (self.condition_dim if self.use_coord_condition else 0)
        dims = [total_input_dim] + [self.hidden_dim] * self.num_blocks + [self.output_dim]
        self.net = create_mlp(
            dims,
            activation_fn=nn.ReLU,
            norm_type=None,
            dropout_p=None,
        )
        self.net.apply(lambda module: init_weights(module, method="kaiming", nonlinearity="relu"))

    def forward(self, inputs, condition_features=None, **kwargs):
        if self.use_coord_condition:
            if condition_features is None:
                raise ValueError("condition_features is required when use_coord_condition=True")
            features = torch.cat([inputs, condition_features], dim=-1)
        else:
            features = inputs
        return self.net(features)


class Stage2GridFeatureMLP(nn.Module):
    """Unified stage2 grid MLP entry for residual/plain and coord on/off ablations."""

    def __init__(
            self,
            input_dim,
            condition_dim,
            hidden_dim=512,
            num_blocks=1,
            output_dim=None,
            arch="residual_cond",
            use_coord_condition=True,
    ):
        super().__init__()
        arch = str(arch).lower()
        if arch not in {"residual_cond", "plain_mlp"}:
            raise ValueError(f"Unsupported grid_mlp_arch: {arch}")

        self.arch = arch
        self.use_coord_condition = bool(use_coord_condition)
        self.input_dim = int(input_dim)
        self.condition_dim = int(condition_dim) if self.use_coord_condition else 0
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.output_dim = int(output_dim if output_dim is not None else input_dim)

        if self.arch == "residual_cond":
            self.impl = SerialModulatorShallow(
                input_dim=self.input_dim,
                condition_dim=self.condition_dim,
                hidden_dim=self.hidden_dim,
                num_blocks=self.num_blocks,
                output_dim=self.output_dim,
                condition_operator="add",
            )
        else:
            self.impl = PlainGridMLP(
                input_dim=self.input_dim,
                condition_dim=self.condition_dim,
                hidden_dim=self.hidden_dim,
                num_blocks=self.num_blocks,
                output_dim=self.output_dim,
                use_coord_condition=self.use_coord_condition,
            )

    def forward(self, inputs, condition_features=None, **kwargs):
        if not self.use_coord_condition:
            condition_features = None
        return self.impl(inputs=inputs, condition_features=condition_features, **kwargs)
