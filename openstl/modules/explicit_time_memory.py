import torch
from torch import nn


class SpatialRefinement(nn.Module):
    """Lightweight spatial residual block shared across time steps."""

    def __init__(self, channels, mlp_ratio=2):
        super().__init__()
        hidden_channels = channels * mlp_ratio
        self.norm = nn.GroupNorm(1, channels)
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, sequence):
        batch_size, length, channels, height, width = sequence.shape
        frames = sequence.reshape(batch_size * length, channels, height, width)
        residual = self.pointwise(self.depthwise(self.norm(frames)))
        return (frames + residual).reshape(
            batch_size, length, channels, height, width
        )


class HistoryFactorizedBlock(nn.Module):
    """Temporal attention followed by shared spatial and channel refinement."""

    def __init__(self, channels, num_heads, mlp_ratio=2):
        super().__init__()
        hidden_channels = channels * mlp_ratio
        self.temporal_norm = nn.LayerNorm(channels)
        self.temporal_attn = nn.MultiheadAttention(
            channels, num_heads, dropout=0.0, batch_first=True
        )
        self.spatial = SpatialRefinement(channels, mlp_ratio=mlp_ratio)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )

    @staticmethod
    def _to_tokens(sequence):
        batch_size, length, channels, height, width = sequence.shape
        tokens = sequence.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width, length, channels
        )
        return tokens, (batch_size, length, channels, height, width)

    @staticmethod
    def _to_sequence(tokens, shape):
        batch_size, length, channels, height, width = shape
        return tokens.reshape(
            batch_size, height, width, length, channels
        ).permute(0, 3, 4, 1, 2).contiguous()

    def forward(self, memory, key_padding_mask=None):
        tokens, shape = self._to_tokens(memory)
        normalized = self.temporal_norm(tokens)
        attended, _ = self.temporal_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        memory = self._to_sequence(tokens, shape)
        memory = self.spatial(memory)
        tokens, shape = self._to_tokens(memory)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return self._to_sequence(tokens, shape)


class ExplicitTimeMemory(nn.Module):
    """Encode history on an explicit temporal axis without flattening T into C."""

    def __init__(
        self,
        channels,
        num_heads=4,
        num_layers=2,
        context_width=7168,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if context_width < channels:
            raise ValueError("context_width must be at least channels")

        self.channels = channels
        self.input_projection = nn.Linear(channels, channels)
        self.blocks = nn.ModuleList([
            HistoryFactorizedBlock(channels, num_heads)
            for _ in range(num_layers)
        ])
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, context_width),
            nn.GELU(),
            nn.Linear(context_width, context_width),
            nn.GELU(),
            nn.Linear(context_width, channels),
        )
        nn.init.normal_(self.context_mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.context_mlp[-1].bias)

    @staticmethod
    def prepare_history_mask(history_mask, batch_size, length, device):
        if history_mask is None:
            return torch.ones(
                batch_size, length, device=device, dtype=torch.bool
            )
        history_mask = torch.as_tensor(
            history_mask, device=device, dtype=torch.bool
        )
        if history_mask.ndim == 1:
            history_mask = history_mask.unsqueeze(0).expand(batch_size, -1)
        if history_mask.shape != (batch_size, length):
            raise ValueError(
                "history_mask must have shape "
                f"({batch_size}, {length}), got {tuple(history_mask.shape)}"
            )
        if not history_mask.any(dim=1).all():
            raise ValueError("Each sample must contain at least one valid history step")
        return history_mask

    @staticmethod
    def expand_padding_mask(history_mask, height, width):
        batch_size, length = history_mask.shape
        return (~history_mask)[:, None, None].expand(
            batch_size, height, width, length
        ).reshape(batch_size * height * width, length)

    def forward(self, history, time_encoding=None, history_mask=None):
        if history.ndim != 5:
            raise ValueError("history must have shape [B, T, C, H, W]")
        batch_size, length, channels, height, width = history.shape
        if channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {channels}"
            )

        valid = self.prepare_history_mask(
            history_mask, batch_size, length, history.device
        )
        memory = history.permute(0, 1, 3, 4, 2)
        memory = self.input_projection(memory).permute(0, 1, 4, 2, 3)
        if time_encoding is not None:
            if time_encoding.shape != (batch_size, length, channels):
                raise ValueError(
                    "time_encoding must have shape "
                    f"({batch_size}, {length}, {channels})"
                )
            memory = memory + time_encoding[:, :, :, None, None]

        key_padding_mask = self.expand_padding_mask(valid, height, width)
        for block in self.blocks:
            memory = block(memory, key_padding_mask)

        weights = valid[:, :, None, None, None].to(memory.dtype)
        denominator = valid.sum(dim=1, keepdim=True).to(memory.dtype)
        denominator = denominator.mul(height * width).clamp_min(1.0)
        pooled = (memory * weights).sum(dim=(1, 3, 4)) / denominator
        context = self.context_mlp(pooled)
        memory = memory + context[:, None, :, None, None]
        return memory, valid
