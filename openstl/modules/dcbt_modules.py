import torch
from torch import nn


class DWSeparableBasis(nn.Module):

    def __init__(self, channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConditionalBasisBlock(nn.Module):

    def __init__(self, channels, kernels=(3, 7, 11), ffn_ratio=2):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.bases = nn.ModuleList([
            DWSeparableBasis(channels, kernel)
            for kernel in kernels
        ])

        self.coeff = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, len(kernels), 1),
        )

        hidden = channels * ffn_ratio
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x, condition):
        normalized = self.norm1(x)
        weights = self.coeff(condition).softmax(dim=1)

        basis = torch.stack(
            [operation(normalized) for operation in self.bases],
            dim=1,
        )
        mixed = (
            basis * weights.unsqueeze(2)
        ).sum(dim=1)

        x = x + mixed
        return x + self.ffn(self.norm2(x))


class DCBTCorrection(nn.Module):
    """Decoder-compatible correction for [B,T,C,H,W] tensors."""

    def __init__(
            self,
            time_steps,
            channels,
            hid_t,
            n_blocks,
            width_ratio=0.5,
            kernels=(3, 7, 11),
            max_delta_ratio=None):
        super().__init__()

        input_channels = time_steps * channels
        difference_channels = (time_steps - 1) * channels
        width = max(32, int(hid_t * width_ratio))

        self.time_steps = int(time_steps)
        self.channels = int(channels)
        self.width = int(width)
        self.max_delta_ratio = (
            None
            if max_delta_ratio is None
            else float(max_delta_ratio)
        )
        if (
            self.max_delta_ratio is not None
            and self.max_delta_ratio <= 0.0
        ):
            raise ValueError(
                "max_delta_ratio must be positive or None"
            )

        self.micro_proj = nn.Conv2d(
            difference_channels,
            width,
            1,
        )
        self.macro_proj = nn.Conv2d(
            input_channels,
            width,
            1,
        )
        self.macro_h = nn.Conv2d(
            width,
            width,
            (1, 11),
            padding=(0, 5),
            groups=width,
            bias=False,
        )
        self.macro_v = nn.Conv2d(
            width,
            width,
            (11, 1),
            padding=(5, 0),
            groups=width,
            bias=False,
        )
        self.nominal_proj = nn.Conv2d(
            input_channels,
            width,
            1,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 3, width, 1),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            ConditionalBasisBlock(
                width,
                kernels=kernels,
                ffn_ratio=2,
            )
            for _ in range(n_blocks)
        ])

        self.out_proj = nn.Conv2d(
            width,
            input_channels,
            1,
        )

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_history, z_nominal):
        batch, steps, channels, height, width = z_history.shape

        if steps != self.time_steps:
            raise ValueError(
                f"Expected {self.time_steps} steps, got {steps}"
            )
        if channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {channels}"
            )
        if z_nominal.shape != z_history.shape:
            raise ValueError(
                "History and nominal latent shapes must match"
            )

        history = z_history.reshape(
            batch,
            steps * channels,
            height,
            width,
        )
        difference = (
            z_history[:, 1:] - z_history[:, :-1]
        ).reshape(
            batch,
            (steps - 1) * channels,
            height,
            width,
        )
        nominal = z_nominal.reshape(
            batch,
            steps * channels,
            height,
            width,
        )

        micro = self.micro_proj(difference)

        macro_base = self.macro_proj(history)
        macro = (
            self.macro_h(macro_base)
            + self.macro_v(macro_base)
        )

        nominal_feature = self.nominal_proj(nominal)

        condition = self.fuse(torch.cat(
            [micro, macro, nominal_feature],
            dim=1,
        ))

        state = nominal_feature
        for block in self.blocks:
            state = block(state, condition)

        delta = self.out_proj(state).reshape(
            batch,
            steps,
            channels,
            height,
            width,
        )

        if self.max_delta_ratio is not None:
            nominal_norm = (
                z_nominal.float()
                .flatten(1)
                .norm(dim=1)
                .clamp_min(1e-12)
            )
            delta_norm = (
                delta.float()
                .flatten(1)
                .norm(dim=1)
                .clamp_min(1e-12)
            )
            maximum_norm = (
                self.max_delta_ratio * nominal_norm
            )
            scale = torch.minimum(
                torch.ones_like(delta_norm),
                maximum_norm / delta_norm,
            )
            delta = delta * scale.to(
                delta.dtype
            ).view(-1, 1, 1, 1, 1)

        return delta
