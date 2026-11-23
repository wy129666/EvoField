from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from timm.layers import DropPath


def _to_bcthw(x):
    return x.permute(0, 2, 1, 3, 4).contiguous()


def _to_btchw(x):
    return x.permute(0, 2, 1, 3, 4).contiguous()


class StaticContent(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre = nn.Conv3d(channels, channels, 1)
        self.local = nn.Conv3d(
            channels, channels, (1, 5, 5),
            padding=(0, 2, 2),
            groups=channels,
            bias=False,
        )
        self.dilated = nn.Conv3d(
            channels, channels, (1, 7, 7),
            padding=(0, 9, 9),
            dilation=(1, 3, 3),
            groups=channels,
            bias=False,
        )
        self.post = nn.Conv3d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.pre(x))
        return self.post(self.dilated(self.local(x)))


class MotionCondition(nn.Module):
    def __init__(self, channels, groups, num_bases):
        super().__init__()
        self.groups = groups
        self.num_bases = num_bases

        self.temporal_context = nn.Conv3d(
            channels, channels, (3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.spatial_context = nn.Conv3d(
            channels, channels, (1, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
            bias=False,
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(3 * channels, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv3d(
                channels, channels, 3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.Conv3d(channels, groups * num_bases, 1),
        )

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
        d1, d2 = self.differences(x)
        context = self.spatial_context(
            self.temporal_context(x)
        )
        logits = self.fuse(
            torch.cat([d1, d2, context], dim=1)
        )
        b, _, t, h, w = logits.shape
        return logits.view(
            b, self.groups, self.num_bases, t, h, w
        )


class SeparableSTBasis(nn.Module):
    def __init__(
        self,
        channels,
        temporal_kernel,
        spatial_kernel,
    ):
        super().__init__()

        temporal_padding = temporal_kernel // 2
        spatial_padding = spatial_kernel // 2

        self.temporal = nn.Conv3d(
            channels, channels,
            (temporal_kernel, 1, 1),
            padding=(temporal_padding, 0, 0),
            groups=channels,
            bias=False,
        )
        self.horizontal = nn.Conv3d(
            channels, channels,
            (1, 1, spatial_kernel),
            padding=(0, 0, spatial_padding),
            groups=channels,
            bias=False,
        )
        self.vertical = nn.Conv3d(
            channels, channels,
            (1, spatial_kernel, 1),
            padding=(0, spatial_padding, 0),
            groups=channels,
            bias=False,
        )
        self.mix = nn.Conv3d(
            channels, channels, 1, bias=False
        )

    def forward(self, x):
        x = self.temporal(x)
        x = self.vertical(self.horizontal(x))
        return self.mix(x)


class MotionConditionedLocalOperator(nn.Module):
    def __init__(
        self,
        channels,
        groups=8,
        basis_specs=((3, 3), (3, 7), (5, 5), (5, 11)),
    ):
        super().__init__()

        if channels % groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible "
                f"by groups={groups}"
            )

        self.channels = channels
        self.groups = groups
        self.num_bases = len(basis_specs)

        self.static = StaticContent(channels)
        self.condition = MotionCondition(
            channels, groups, self.num_bases
        )
        self.bases = nn.ModuleList([
            SeparableSTBasis(channels, kt, ks)
            for kt, ks in basis_specs
        ])
        self.out = nn.Conv3d(channels, channels, 1)

    def forward(self, x, return_routing=False):
        static = self.static(x)
        weights = self.condition(x).softmax(dim=2)

        basis = torch.stack(
            [operator(static) for operator in self.bases],
            dim=1,
        )

        batch, count, channels, time, height, width = (
            basis.shape
        )
        channels_per_group = channels // self.groups

        basis = basis.view(
            batch,
            count,
            self.groups,
            channels_per_group,
            time,
            height,
            width,
        )
        basis = basis.permute(
            0, 2, 1, 3, 4, 5, 6
        ).contiguous()

        mixed = (
            basis * weights.unsqueeze(3)
        ).sum(dim=2)

        mixed = mixed.reshape(
            batch, channels, time, height, width
        )
        output = self.out(mixed)

        if return_routing:
            return output, weights
        return output


class FoldedTimeChannelFFN(nn.Module):
    def __init__(
        self,
        time_steps,
        channels,
        ratio=6.0,
        drop=0.0,
        drop_path=0.0,
        init_value=1e-2,
    ):
        super().__init__()

        self.time_steps = time_steps
        self.channels = channels

        total = time_steps * channels
        hidden = int(total * ratio)

        self.norm = nn.BatchNorm2d(total)
        self.fc1 = nn.Conv2d(total, hidden, 1)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden, total, 1)
        self.drop2 = nn.Dropout(drop)

        self.scale = nn.Parameter(
            init_value * torch.ones(total)
        )
        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

    def forward(self, x):
        batch, channels, time, height, width = x.shape

        if (
            time != self.time_steps
            or channels != self.channels
        ):
            raise ValueError(
                f"expected C,T={self.channels},"
                f"{self.time_steps}; got {channels},{time}"
            )

        folded = _to_btchw(x).reshape(
            batch,
            time * channels,
            height,
            width,
        )
        residual = self.fc2(
            self.drop1(
                self.act(
                    self.fc1(self.norm(folded))
                )
            )
        )
        residual = self.drop2(residual)

        folded = folded + self.drop_path(
            self.scale[:, None, None] * residual
        )

        return _to_bcthw(
            folded.reshape(
                batch,
                time,
                channels,
                height,
                width,
            )
        )


class MLOTAUBlock(nn.Module):
    def __init__(
        self,
        time_steps,
        channels,
        groups=8,
        basis_specs=((3, 3), (3, 7), (5, 5), (5, 11)),
        mlp_ratio=6.0,
        drop=0.0,
        drop_path=0.0,
        init_value=1e-2,
    ):
        super().__init__()

        self.norm = nn.GroupNorm(1, channels)
        self.operator = MotionConditionedLocalOperator(
            channels,
            groups,
            basis_specs,
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
            time_steps,
            channels,
            mlp_ratio,
            drop,
            drop_path,
            init_value,
        )

    def forward(self, x):
        residual = self.operator(self.norm(x))
        x = x + self.drop_path(
            self.scale[None, :, None, None, None]
            * residual
        )
        return self.ffn(x)


class MLOTAUTranslator(nn.Module):
    def __init__(
        self,
        time_steps,
        in_channels,
        channels=80,
        depth=8,
        groups=8,
        basis_specs=((3, 3), (3, 7), (5, 5), (5, 11)),
        mlp_ratio=6.0,
        drop=0.0,
        drop_path=0.0,
        init_value=1e-2,
    ):
        super().__init__()

        if depth < 2:
            raise ValueError(
                "MLO-TAU requires depth >= 2"
            )

        self.time_steps = time_steps
        self.in_channels = in_channels

        self.in_proj = nn.Conv3d(
            in_channels, channels, 1
        )

        drop_rates = torch.linspace(
            0, drop_path, depth
        ).tolist()

        self.blocks = nn.ModuleList([
            MLOTAUBlock(
                time_steps=time_steps,
                channels=channels,
                groups=groups,
                basis_specs=basis_specs,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_rates[index],
                init_value=init_value,
            )
            for index in range(depth)
        ])

        self.out_norm = nn.GroupNorm(1, channels)
        self.out_proj = nn.Conv3d(
            channels, in_channels, 1
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
