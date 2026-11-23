from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from openstl.modules.mlo_tau_modules import (
    StaticContent,
    FoldedTimeChannelFFN,
    _to_bcthw,
    _to_btchw,
)
from openstl.modules.mlo_tau_v2_modules import (
    DIRECTION_SPECS,
    IdentityTransport,
    DirectionalTransportBasis,
    HorizonMotionCondition,
)


class HorizonMotionConditionWithContext(HorizonMotionCondition):
    """V2 router with access to its pre-logit condition feature."""

    def forward(self, x, return_context=False):
        batch, _, time, height, width = x.shape

        if time != self.time_steps:
            raise ValueError(
                f"expected time={self.time_steps}; got {time}"
            )

        d1, d2 = self.differences(x)

        content = x + self.spatial_context(
            self.temporal_context(x)
        )

        future = self.future_slot_embedding.expand(
            batch, -1, -1, height, width
        )

        context = self.fuse(
            torch.cat([d1, d2, content, future], dim=1)
        )
        logits = self.route_out(context)

        logits = logits.view(
            batch,
            self.groups,
            self.num_bases,
            time,
            height,
            width,
        )

        if return_context:
            return logits, context

        return logits


class SignedLocalSourceResidual(nn.Module):
    """Signed local creation/removal term independent of softmax routing."""

    def __init__(self, channels, hidden_channels=None):
        super().__init__()

        channels = int(channels)
        hidden = (
            max(8, channels // 2)
            if hidden_channels is None
            else int(hidden_channels)
        )

        self.reduce = nn.Conv3d(
            4 * channels,
            hidden,
            kernel_size=1,
            bias=False,
        )
        self.depthwise = nn.Conv3d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
            bias=False,
        )
        self.act = nn.GELU()
        self.restore = nn.Conv3d(
            hidden,
            channels,
            kernel_size=1,
            bias=True,
        )

        # Deliberately not zero-initialized. Stability is provided by
        # the existing block LayerScale initialized to 1e-2.

    def forward(self, x, route_context):
        d1, d2 = HorizonMotionCondition.differences(x)

        blur = F.avg_pool3d(
            x,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
            count_include_pad=False,
        )
        high_frequency = x - blur

        features = torch.cat(
            [high_frequency, d1, d2, route_context],
            dim=1,
        )

        return self.restore(
            self.act(
                self.depthwise(
                    self.reduce(features)
                )
            )
        )


class TransportSourceOperator(nn.Module):
    """Nine-way V2 transport plus an independent signed source term."""

    def __init__(
        self,
        channels,
        time_steps,
        groups=8,
        source_hidden_channels=None,
    ):
        super().__init__()

        if channels % groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible "
                f"by groups={groups}"
            )

        self.channels = int(channels)
        self.groups = int(groups)
        self.num_bases = 9

        self.basis_names = tuple(
            name for name, _, _ in DIRECTION_SPECS
        )

        self.static_content = StaticContent(channels)

        bases = [IdentityTransport()]
        bases.extend(
            DirectionalTransportBasis(channels, dy, dx)
            for _, dy, dx in DIRECTION_SPECS[1:]
        )
        self.bases = nn.ModuleList(bases)

        self.condition = HorizonMotionConditionWithContext(
            channels=channels,
            groups=groups,
            num_bases=self.num_bases,
            time_steps=time_steps,
        )

        self.transport_out = nn.Conv3d(
            channels,
            channels,
            kernel_size=1,
        )

        self.source = SignedLocalSourceResidual(
            channels=channels,
            hidden_channels=source_hidden_channels,
        )

    def forward(self, x, return_aux=False):
        static = self.static_content(x)

        logits, route_context = self.condition(
            x,
            return_context=True,
        )
        weights = logits.softmax(dim=2)

        # Memory-equivalent streaming mixture.  This computes
        # exactly sum_k p_k O_k(x) without materializing the full
        # [B, K, C, T, H, W] candidate tensor.
        batch, channels, time, height, width = static.shape
        channels_per_group = channels // self.groups

        mixed = None
        for basis_index, basis in enumerate(self.bases):
            candidate = basis(static).reshape(
                batch,
                self.groups,
                channels_per_group,
                time,
                height,
                width,
            )
            contribution = (
                candidate
                * weights[:, :, basis_index].unsqueeze(2)
            )
            mixed = (
                contribution
                if mixed is None
                else mixed + contribution
            )

        mixed = mixed.reshape(
            batch,
            channels,
            time,
            height,
            width,
        )

        transport = self.transport_out(mixed)
        source = self.source(x, route_context)
        output = transport + source

        if return_aux:
            return output, {
                "routing": weights,
                "transport": transport,
                "source": source,
                "route_context": route_context,
            }

        return output


class MLOTAUV3TSRBlock(nn.Module):
    def __init__(
        self,
        time_steps,
        channels,
        groups=8,
        mlp_ratio=6.0,
        drop=0.0,
        drop_path=0.0,
        init_value=1e-2,
        source_hidden_channels=None,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(1, channels)
        self.operator = TransportSourceOperator(
            channels=channels,
            time_steps=time_steps,
            groups=groups,
            source_hidden_channels=source_hidden_channels,
        )

        self.scale = nn.Parameter(
            init_value * torch.ones(channels)
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

        self.ffn = FoldedTimeChannelFFN(
            time_steps=time_steps,
            channels=channels,
            ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            init_value=init_value,
        )

    def forward(self, x):
        residual = self.operator(self.norm(x))
        x = x + self.drop_path(
            self.scale[None, :, None, None, None]
            * residual
        )
        return self.ffn(x)


class MLOTAUV3TSRTranslator(nn.Module):
    def __init__(
        self,
        time_steps,
        in_channels,
        channels=80,
        depth=8,
        groups=8,
        mlp_ratio=6.0,
        drop=0.0,
        drop_path=0.0,
        init_value=1e-2,
        source_hidden_channels=None,
    ):
        super().__init__()

        if depth < 2:
            raise ValueError("MLO-TAU V3-TSR requires depth >= 2")

        self.time_steps = int(time_steps)
        self.in_channels = int(in_channels)

        self.in_proj = nn.Conv3d(
            in_channels,
            channels,
            kernel_size=1,
        )

        drop_rates = torch.linspace(
            0,
            drop_path,
            depth,
        ).tolist()

        self.blocks = nn.ModuleList([
            MLOTAUV3TSRBlock(
                time_steps=time_steps,
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_rates[index],
                init_value=init_value,
                source_hidden_channels=source_hidden_channels,
            )
            for index in range(depth)
        ])

        self.out_norm = nn.GroupNorm(1, channels)
        self.out_proj = nn.Conv3d(
            channels,
            in_channels,
            kernel_size=1,
        )

    def forward(self, x):
        batch, time, channels, _, _ = x.shape

        if (
            time != self.time_steps
            or channels != self.in_channels
        ):
            raise ValueError(
                f"expected T,C={self.time_steps},"
                f"{self.in_channels}; got {time},{channels}"
            )

        x = self.in_proj(_to_bcthw(x))

        for block in self.blocks:
            x = block(x)

        x = self.out_proj(self.out_norm(x))
        return _to_btchw(x)
