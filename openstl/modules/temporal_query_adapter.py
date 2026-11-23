import math

import torch
from torch import nn


class RelativeTimeEncoding(nn.Module):
    """Parameter-free sinusoidal encoding for scalar relative timestamps."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        half = (channels + 1) // 2
        denominator = max(half - 1, 1)
        inv_freq = torch.exp(
            -math.log(10000.0) * torch.arange(half).float() / denominator
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, timestamps):
        angles = timestamps.float().unsqueeze(-1) * self.inv_freq
        encoding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return encoding[..., :self.channels].to(dtype=timestamps.dtype)


class TemporalQueryBlock(nn.Module):
    """Temporal cross-attention shared over all spatial locations."""

    def __init__(self, channels, num_heads, mlp_ratio=2):
        super().__init__()
        hidden_channels = channels * mlp_ratio
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )

    def forward(self, query, memory, key_padding_mask=None):
        normalized_query = self.query_norm(query)
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attn(
            normalized_query,
            normalized_memory,
            normalized_memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))


class TemporalQueryAdapter(nn.Module):
    """Zero-gated temporal-query residual adapter for SimVP latents."""

    def __init__(self, channels, num_heads=4, num_layers=1, scale_init=0.0,
                 use_time=True, max_future_steps=20):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.time_encoding = RelativeTimeEncoding(channels)
        self.use_time = bool(use_time)
        self.max_future_steps = int(max_future_steps)
        if not self.use_time:
            if self.max_future_steps < 1:
                raise ValueError("max_future_steps must be at least 1")
            self.future_slots = nn.Parameter(
                torch.zeros(self.max_future_steps, channels)
            )
            nn.init.trunc_normal_(self.future_slots, std=0.02)
        self.memory_projection = nn.Linear(channels, channels)
        self.query_projection = nn.Linear(channels, channels)
        self.blocks = nn.ModuleList([
            TemporalQueryBlock(channels, num_heads)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(channels, channels)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    @staticmethod
    def _prepare_timestamp(
        timestamp, batch_size, length, device, dtype, default_start
    ):
        if timestamp is None:
            timestamp = torch.arange(
                default_start,
                default_start + length,
                device=device,
                dtype=dtype,
            )
        else:
            timestamp = torch.as_tensor(
                timestamp, device=device, dtype=dtype
            )

        if timestamp.ndim == 1:
            if timestamp.numel() != length:
                raise ValueError(
                    f"Expected {length} timestamps, got {timestamp.numel()}"
                )
            timestamp = timestamp.unsqueeze(0).expand(batch_size, -1)
        elif timestamp.ndim == 2:
            if timestamp.shape != (batch_size, length):
                raise ValueError(
                    "Timestamp shape must be "
                    f"({batch_size}, {length}), got {tuple(timestamp.shape)}"
                )
        else:
            raise ValueError("Timestamps must have shape [T] or [B, T]")
        return timestamp

    def forward(
        self,
        z_base,
        e_hist,
        t_hist=None,
        t_future=None,
        history_mask=None,
    ):
        if z_base.ndim != 5 or e_hist.ndim != 5:
            raise ValueError("z_base and e_hist must have shape [B, T, C, H, W]")

        batch_size, future_length, channels, height, width = z_base.shape
        hist_batch, history_length, hist_channels, hist_h, hist_w = e_hist.shape
        if (hist_batch, hist_channels, hist_h, hist_w) != (
            batch_size, channels, height, width
        ):
            raise ValueError("History and future latent dimensions do not match")

        memory = e_hist.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width, history_length, channels
        )
        query = z_base.permute(0, 3, 4, 1, 2).reshape(
            batch_size * height * width, future_length, channels
        )

        memory = self.memory_projection(memory)
        query = self.query_projection(query)
        if self.use_time:
            t_hist = self._prepare_timestamp(
                t_hist, batch_size, history_length,
                z_base.device, z_base.dtype, 1 - history_length,
            )
            t_future = self._prepare_timestamp(
                t_future, batch_size, future_length,
                z_base.device, z_base.dtype, 1,
            )
            anchor = t_hist[:, -1:]
            t_hist = t_hist - anchor
            t_future = t_future - anchor
            normalizer = torch.cat(
                [t_hist.abs(), t_future.abs()], dim=1
            ).amax(dim=1, keepdim=True).clamp_min(1.0)
            t_hist = t_hist / normalizer
            t_future = t_future / normalizer

            hist_time = self.time_encoding(t_hist)
            future_time = self.time_encoding(t_future)
            hist_time = hist_time[:, None, None].expand(
                batch_size, height, width, history_length, channels
            ).reshape(batch_size * height * width, history_length, channels)
            future_time = future_time[:, None, None].expand(
                batch_size, height, width, future_length, channels
            ).reshape(batch_size * height * width, future_length, channels)
            memory = memory + hist_time
            query = query + future_time
        else:
            if future_length > self.max_future_steps:
                raise ValueError(
                    f"future_length={future_length} exceeds "
                    f"max_future_steps={self.max_future_steps}"
                )
            slots = self.future_slots[:future_length]
            slots = slots[None, None, None].expand(
                batch_size, height, width, future_length, channels
            ).reshape(batch_size * height * width, future_length, channels)
            query = query + slots

        key_padding_mask = None
        if history_mask is not None:
            history_mask = torch.as_tensor(
                history_mask, device=z_base.device, dtype=torch.bool
            )
            if history_mask.ndim == 1:
                history_mask = history_mask.unsqueeze(0).expand(batch_size, -1)
            if history_mask.shape != (batch_size, history_length):
                raise ValueError(
                    "history_mask must have shape "
                    f"({batch_size}, {history_length})"
                )
            key_padding_mask = (~history_mask)[:, None, None].expand(
                batch_size, height, width, history_length
            ).reshape(batch_size * height * width, history_length)

        for block in self.blocks:
            query = block(query, memory, key_padding_mask)

        delta = self.output_projection(query)
        delta = delta.reshape(
            batch_size, height, width, future_length, channels
        ).permute(0, 3, 4, 1, 2).contiguous()
        return z_base + self.scale * delta
