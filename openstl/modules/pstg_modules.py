import torch
from torch import nn

from .simvp_modules import AttentionModule


class PredictiveSpatioTemporalGate(nn.Module):
    """Spatial gating followed by explicit temporal routing."""

    def __init__(
        self,
        channels,
        use_spatial_gate=True,
        use_temporal_gate=True,
        scale_init=1e-2,
    ):
        super().__init__()

        if not use_spatial_gate and not use_temporal_gate:
            raise ValueError("At least one PSTG branch must be enabled.")

        self.channels = int(channels)
        self.use_spatial_gate = bool(use_spatial_gate)
        self.use_temporal_gate = bool(use_temporal_gate)

        if self.use_spatial_gate:
            # Reuse the mature gSTA large-kernel spatial attention.
            self.spatial_gate = AttentionModule(
                self.channels,
                kernel_size=21,
                dilation=3,
            )
        else:
            self.spatial_gate = None

        if self.use_temporal_gate:
            # GAP descriptors retain the explicit time axis: [B,C,T].
            self.temporal_gate = nn.Sequential(
                nn.Conv1d(
                    self.channels,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    groups=self.channels,
                ),
                nn.GELU(),
                nn.Conv1d(
                    self.channels,
                    self.channels,
                    kernel_size=1,
                ),
            )
        else:
            self.temporal_gate = None

        # Non-zero LayerScale as preregistered.
        self.layer_scale = nn.Parameter(
            torch.full((1, 1, self.channels, 1, 1), float(scale_init))
        )

    def forward(self, z):
        if z.ndim != 5:
            raise ValueError(
                "PSTG expects [B,T,C,H,W], got {}".format(tuple(z.shape))
            )

        b, t, c, h, w = z.shape

        if c != self.channels:
            raise ValueError(
                "Expected {} channels, got {}".format(self.channels, c)
            )

        if self.use_spatial_gate:
            spatial = self.spatial_gate(
                z.reshape(b * t, c, h, w)
            ).reshape(b, t, c, h, w)
        else:
            spatial = z

        if self.use_temporal_gate:
            descriptor = z.mean(dim=(-1, -2)).transpose(1, 2)
            temporal = torch.sigmoid(
                self.temporal_gate(descriptor)
            ).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        else:
            temporal = 1.0

        return z + self.layer_scale * temporal * spatial
