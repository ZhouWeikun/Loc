import torch.nn as nn

from models.cond_modulator_shallow_serial import SerialModulatorShallow


class Stage2GridFeatureMLP(nn.Module):
    """Minimal Stage-2 grid MLP: residual conditional projection only."""

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
        if arch != "residual_cond":
            raise ValueError(f"Minimal pipeline only supports grid_mlp_arch='residual_cond', got {arch!r}")

        self.arch = arch
        self.use_coord_condition = bool(use_coord_condition)
        self.input_dim = int(input_dim)
        self.condition_dim = int(condition_dim) if self.use_coord_condition else 0
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.output_dim = int(output_dim if output_dim is not None else input_dim)

        self.impl = SerialModulatorShallow(
            input_dim=self.input_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            output_dim=self.output_dim,
            condition_operator="add",
        )

    def forward(self, inputs, condition_features=None, **kwargs):
        if not self.use_coord_condition:
            condition_features = None
        return self.impl(inputs=inputs, condition_features=condition_features, **kwargs)
