from __future__ import annotations

import torch
import torch.nn as nn
from timm.layers import DropPath

from openstl.modules.mlo_tau_modules import (
    StaticContent,
    FoldedTimeChannelFFN,
    _to_bcthw,
    _to_btchw,
)


DIRECTION_SPECS = (
    ("static", 0, 0),
    ("left", 0, -1),
    ("right", 0, 1),
    ("up", -1, 0),
    ("down", 1, 0),
    ("up_left", -1, -1),
    ("up_right", -1, 1),
    ("down_left", 1, -1),
    ("down_right", 1, 1),
)


class IdentityTransport(nn.Module):
    def forward(self, x):
        return x


class DirectionalTransportBasis(nn.Module):
    """Learnable DW 3x3 transport initialized as an exact shift."""

    def __init__(self, channels, delta_y, delta_x):
        super().__init__()

        if delta_y not in {-1, 0, 1}:
            raise ValueError(delta_y)
        if delta_x not in {-1, 0, 1}:
            raise ValueError(delta_x)

        self.delta_y = int(delta_y)
        self.delta_x = int(delta_x)

        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.conv.weight.zero_()

            # Conv3d uses cross-correlation. To move an input impulse
            # toward (+dy,+dx), sample from the opposite kernel offset.
            kernel_y = 1 - self.delta_y
            kernel_x = 1 - self.delta_x
            self.conv.weight[:, 0, 0, kernel_y, kernel_x] = 1.0

    def forward(self, x):
        return self.conv(x)


class AppearanceChangeBasis(nn.Module):
    """Local appearance operator, initialized as identity."""

    def __init__(self, channels):
        super().__init__()

        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
            bias=False,
        )
        self.mix = nn.Conv3d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.depthwise.weight.zero_()
            self.depthwise.weight[:, 0, 0, 1, 1] = 1.0

            self.mix.weight.zero_()
            channels = self.mix.weight.shape[0]
            indices = torch.arange(channels)
            self.mix.weight[indices, indices, 0, 0, 0] = 1.0

    def forward(self, x):
        return self.mix(self.depthwise(x))


class HorizonMotionCondition(nn.Module):
    """Route using X, first/second differences and future-slot encoding."""

    def __init__(
        self,
        channels,
        groups,
        num_bases,
        time_steps,
    ):
        super().__init__()

        self.channels = int(channels)
        self.groups = int(groups)
        self.num_bases = int(num_bases)
        self.time_steps = int(time_steps)

        self.temporal_context = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.spatial_context = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
            bias=False,
        )

        self.future_slot_embedding = nn.Parameter(
            torch.empty(1, channels, time_steps, 1, 1)
        )
        nn.init.normal_(
            self.future_slot_embedding,
            mean=0.0,
            std=0.02,
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(
                4 * channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GELU(),
        )
        self.route_out = nn.Conv3d(
            channels,
            groups * num_bases,
            kernel_size=1,
        )

        # Start close to uniform routing without blocking gradients.
        nn.init.normal_(
            self.route_out.weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.route_out.bias)

    @staticmethod
    def differences(x):
        d1 = torch.zeros_like(x)
        d1[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]

        d2 = torch.zeros_like(x)
        d2[:, :, 2:] = (
            x[:, :, 2:]
            - 2.0 * x[:, :, 1:-1]
            + x[:, :, :-2]
        )
        return d1, d2

    def forward(self, x):
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

        fused = self.fuse(
            torch.cat([d1, d2, content, future], dim=1)
        )
        logits = self.route_out(fused)

        return logits.view(
            batch,
            self.groups,
            self.num_bases,
            time,
            height,
            width,
        )


class DirectionalTransportOperator(nn.Module):
    """Ten-way future-conditioned transport competition."""

    def __init__(
        self,
        channels,
        time_steps,
        groups=8,
    ):
        super().__init__()

        if channels % groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible "
                f"by groups={groups}"
            )

        self.channels = int(channels)
        self.groups = int(groups)
        self.num_bases = 10

        self.basis_names = (
            "static",
            "left",
            "right",
            "up",
            "down",
            "up_left",
            "up_right",
            "down_left",
            "down_right",
            "appearance",
        )

        self.static_content = StaticContent(channels)

        bases = [IdentityTransport()]
        bases.extend(
            DirectionalTransportBasis(channels, dy, dx)
            for _, dy, dx in DIRECTION_SPECS[1:]
        )
        bases.append(AppearanceChangeBasis(channels))
        self.bases = nn.ModuleList(bases)

        self.condition = HorizonMotionCondition(
            channels=channels,
            groups=groups,
            num_bases=self.num_bases,
            time_steps=time_steps,
        )
        self.out = nn.Conv3d(
            channels,
            channels,
            kernel_size=1,
        )

    def forward(self, x, return_routing=False):
        static = self.static_content(x)

        weights = self.condition(x).softmax(dim=2)

        candidates = torch.stack(
            [basis(static) for basis in self.bases],
            dim=1,
        )

        batch, count, channels, time, height, width = (
            candidates.shape
        )
        channels_per_group = channels // self.groups

        candidates = candidates.view(
            batch,
            count,
            self.groups,
            channels_per_group,
            time,
            height,
            width,
        )
        candidates = candidates.permute(
            0, 2, 1, 3, 4, 5, 6
        ).contiguous()

        mixed = (
            candidates * weights.unsqueeze(3)
        ).sum(dim=2)

        mixed = mixed.reshape(
            batch,
            channels,
            time,
            height,
            width,
        )
        output = self.out(mixed)

        if return_routing:
            return output, weights

        return output


class MLOTAUV2Block(nn.Module):
    def __init__(
        self,
        time_steps,
        channels,
        groups=8,
        mlp_ratio=6.0,
        drop=0.0,
        drop_path=0.0,
        init_value=1e-2,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(1, channels)
        self.operator = DirectionalTransportOperator(
            channels=channels,
            time_steps=time_steps,
            groups=groups,
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


class MLOTAUV2Translator(nn.Module):
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
    ):
        super().__init__()

        if depth < 2:
            raise ValueError("MLO-TAU V2 requires depth >= 2")

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
            MLOTAUV2Block(
                time_steps=time_steps,
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_rates[index],
                init_value=init_value,
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
