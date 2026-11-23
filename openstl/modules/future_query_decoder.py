import torch
from torch import nn

from openstl.modules.explicit_time_memory import SpatialRefinement
from openstl.modules.temporal_query_adapter import RelativeTimeEncoding


class FutureQueryBlock(nn.Module):
    """Cross-attend future queries to history at every spatial position."""

    def __init__(self, channels, num_heads, mlp_ratio=2):
        super().__init__()
        hidden_channels = channels * mlp_ratio
        self.query_norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.cross_attn = nn.MultiheadAttention(
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

    def forward(self, query, memory, key_padding_mask=None):
        query_tokens, query_shape = self._to_tokens(query)
        memory_tokens, _ = self._to_tokens(memory)
        attended, _ = self.cross_attn(
            self.query_norm(query_tokens),
            self.memory_norm(memory_tokens),
            self.memory_norm(memory_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query_tokens = query_tokens + attended
        query = self._to_sequence(query_tokens, query_shape)
        query = self.spatial(query)
        query_tokens, query_shape = self._to_tokens(query)
        query_tokens = query_tokens + self.ffn(self.ffn_norm(query_tokens))
        return self._to_sequence(query_tokens, query_shape)


class FutureQueryDecoder(nn.Module):
    """Decode a variable number of learned future queries from history memory."""

    def __init__(
        self,
        channels,
        num_heads=4,
        num_layers=2,
        max_future_steps=20,
    ):
        super().__init__()
        if max_future_steps < 1:
            raise ValueError("max_future_steps must be at least 1")
        self.channels = channels
        self.max_future_steps = max_future_steps
        self.future_slots = nn.Parameter(
            torch.empty(max_future_steps, channels)
        )
        nn.init.trunc_normal_(self.future_slots, std=0.02)
        self.blocks = nn.ModuleList([
            FutureQueryBlock(channels, num_heads)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(channels, channels)

    def prepare_slot_indices(
        self, slot_indices, batch_size, future_length, device
    ):
        if slot_indices is None:
            slot_indices = torch.arange(future_length, device=device)
        else:
            slot_indices = torch.as_tensor(
                slot_indices, device=device, dtype=torch.long
            )
        if slot_indices.ndim == 1:
            if slot_indices.numel() != future_length:
                raise ValueError(
                    f"Expected {future_length} slot indices, "
                    f"got {slot_indices.numel()}"
                )
            slot_indices = slot_indices.unsqueeze(0).expand(batch_size, -1)
        elif slot_indices.shape != (batch_size, future_length):
            raise ValueError(
                "slot_indices must have shape "
                f"({batch_size}, {future_length})"
            )
        if slot_indices.min() < 0 or slot_indices.max() >= self.max_future_steps:
            raise ValueError(
                f"slot_indices must be in [0, {self.max_future_steps - 1}]"
            )
        return slot_indices

    def forward(
        self,
        memory,
        history_mask,
        future_length,
        future_time_encoding=None,
        slot_indices=None,
    ):
        batch_size, _, channels, height, width = memory.shape
        if not 1 <= future_length <= self.max_future_steps:
            raise ValueError(
                f"future_length must be in [1, {self.max_future_steps}]"
            )
        slot_indices = self.prepare_slot_indices(
            slot_indices,
            batch_size,
            future_length,
            memory.device,
        )
        query = self.future_slots[slot_indices]
        if future_time_encoding is not None:
            if future_time_encoding.shape != (
                batch_size, future_length, channels
            ):
                raise ValueError(
                    "future_time_encoding must have shape "
                    f"({batch_size}, {future_length}, {channels})"
                )
            query = query + future_time_encoding
        query = query[:, :, :, None, None].expand(
            batch_size, future_length, channels, height, width
        ).contiguous()

        key_padding_mask = (~history_mask)[:, None, None].expand(
            batch_size, height, width, history_mask.shape[1]
        ).reshape(batch_size * height * width, history_mask.shape[1])
        for block in self.blocks:
            query = block(query, memory, key_padding_mask)

        query = query.permute(0, 1, 3, 4, 2)
        query = self.output_projection(query)
        return query.permute(0, 1, 4, 2, 3).contiguous()


class FullTemporalQuery(nn.Module):
    """Complete temporal-query middle network shared by F0 and F1."""

    def __init__(
        self,
        channels,
        memory_encoder,
        num_heads=4,
        num_layers=2,
        max_future_steps=20,
        use_time=False,
    ):
        super().__init__()
        self.channels = channels
        self.use_time = bool(use_time)
        self.time_encoding = RelativeTimeEncoding(channels)
        self.memory_encoder = memory_encoder
        self.query_decoder = FutureQueryDecoder(
            channels=channels,
            num_heads=num_heads,
            num_layers=num_layers,
            max_future_steps=max_future_steps,
        )

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
        elif timestamp.shape != (batch_size, length):
            raise ValueError(
                "Timestamp shape must be "
                f"({batch_size}, {length}), got {tuple(timestamp.shape)}"
            )
        return timestamp

    def _time_features(
        self,
        history,
        future_length,
        t_hist=None,
        t_future=None,
    ):
        batch_size, history_length = history.shape[:2]
        t_hist = self._prepare_timestamp(
            t_hist,
            batch_size,
            history_length,
            history.device,
            history.dtype,
            1 - history_length,
        )
        t_future = self._prepare_timestamp(
            t_future,
            batch_size,
            future_length,
            history.device,
            history.dtype,
            1,
        )
        anchor = t_hist[:, -1:]
        t_hist = t_hist - anchor
        t_future = t_future - anchor
        normalizer = torch.cat(
            [t_hist.abs(), t_future.abs()], dim=1
        ).amax(dim=1, keepdim=True).clamp_min(1.0)
        return (
            self.time_encoding(t_hist / normalizer),
            self.time_encoding(t_future / normalizer),
        )

    def forward(
        self,
        history,
        future_length=None,
        t_hist=None,
        t_future=None,
        history_mask=None,
        slot_indices=None,
    ):
        if history.ndim != 5:
            raise ValueError("history must have shape [B, T, C, H, W]")
        batch_size, history_length = history.shape[:2]
        if future_length is None:
            if t_future is None:
                future_length = history_length
            else:
                future_length = torch.as_tensor(t_future).shape[-1]
        future_length = int(future_length)

        history_time = None
        future_time = None
        if self.use_time:
            history_time, future_time = self._time_features(
                history,
                future_length,
                t_hist=t_hist,
                t_future=t_future,
            )

        memory, valid = self.memory_encoder(
            history,
            time_encoding=history_time,
            history_mask=history_mask,
        )
        return self.query_decoder(
            memory,
            history_mask=valid,
            future_length=future_length,
            future_time_encoding=future_time,
            slot_indices=slot_indices,
        )
