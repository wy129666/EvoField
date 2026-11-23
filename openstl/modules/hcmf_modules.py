import torch
import torch.nn as nn

from .simvp_modules import gInception_ST


class HorizonConditionedInception(gInception_ST):
    """Horizon-conditioned fusion of existing IncepU scale branches."""

    def __init__(
        self,
        C_in,
        C_hid,
        C_out,
        num_horizons,
        hidden_dim=16,
        incep_ker=(3, 5, 7, 11),
        groups=8,
    ):
        super().__init__(
            C_in=C_in,
            C_hid=C_hid,
            C_out=C_out,
            incep_ker=list(incep_ker),
            groups=groups,
        )

        if C_out % num_horizons != 0:
            raise ValueError(
                f"C_out={C_out} must be divisible by "
                f"num_horizons={num_horizons}"
            )

        self.num_horizons = int(num_horizons)
        self.channels_per_horizon = C_out // num_horizons
        self.num_scales = len(self.layers)

        self.horizon_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_scales),
        )

        # Uniform fusion at initialization:
        # num_scales * softmax(0) = 1 for every branch.
        nn.init.zeros_(self.horizon_mlp[-1].weight)
        nn.init.zeros_(self.horizon_mlp[-1].bias)

        positions = torch.linspace(
            1.0 / self.num_horizons,
            1.0,
            self.num_horizons,
        ).unsqueeze(-1)

        self.register_buffer(
            "horizon_positions",
            positions,
            persistent=True,
        )

    def scale_weights(self):
        logits = self.horizon_mlp(self.horizon_positions)
        return self.num_scales * torch.softmax(logits, dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        weights = self.scale_weights()

        batch, _, height, width = x.shape
        output = 0

        # Preserve the original branch-addition order.
        for scale_index, layer in enumerate(self.layers):
            branch = layer(x)
            out_channels = branch.shape[1]

            branch = branch.reshape(
                batch,
                self.num_horizons,
                self.channels_per_horizon,
                height,
                width,
            )

            weight = weights[:, scale_index].reshape(
                1,
                self.num_horizons,
                1,
                1,
                1,
            )

            branch = (branch * weight).reshape(
                batch,
                out_channels,
                height,
                width,
            )
            output = output + branch

        return output
