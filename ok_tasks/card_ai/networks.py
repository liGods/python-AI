from __future__ import annotations

from typing import Any

from ok_tasks.card_ai.features import (
    HISTORY_FEATURE_SIZE,
    HISTORY_LIMIT,
    STATIC_FEATURE_SIZE,
    TEACHER_FEATURE_SIZE,
)


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("高强度训练需要在 WSL2 训练环境安装 PyTorch") from error
    return torch, nn


def build_student(backbone: str = "lstm", width: int = 384) -> Any:
    torch, nn = require_torch()

    class ResidualBlock(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.block = nn.Sequential(nn.Linear(size, size), nn.LayerNorm(size), nn.GELU(), nn.Linear(size, size))

        def forward(self, value):
            return torch.nn.functional.gelu(value + self.block(value))

    class ActionRanker(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone_name = backbone
            self.static_encoder = nn.Sequential(
                nn.Linear(STATIC_FEATURE_SIZE, width), nn.LayerNorm(width), nn.GELU(), ResidualBlock(width)
            )
            if backbone == "lstm":
                self.history_encoder = nn.LSTM(HISTORY_FEATURE_SIZE, width // 2, num_layers=2, batch_first=True)
                history_width = width // 2
            elif backbone == "transformer":
                self.history_projection = nn.Linear(HISTORY_FEATURE_SIZE, width)
                self.position_embedding = nn.Parameter(torch.zeros(1, HISTORY_LIMIT, width))
                layer = nn.TransformerEncoderLayer(
                    d_model=width, nhead=8, dim_feedforward=width * 4, dropout=0.1, activation="gelu", batch_first=True
                )
                self.history_encoder = nn.TransformerEncoder(layer, num_layers=4)
                history_width = width
            else:
                raise ValueError(f"未知历史编码器: {backbone}")
            self.fusion = nn.Sequential(
                nn.Linear(width + history_width, width), nn.LayerNorm(width), nn.GELU(), ResidualBlock(width), ResidualBlock(width)
            )
            self.role_heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(3)])
            self.opponent_head = nn.Linear(width, 32)
            self.teammate_head = nn.Linear(width, 1)

        def forward(self, static, history, history_mask, role_index):
            static_value = self.static_encoder(static)
            if self.backbone_name == "lstm":
                output, _ = self.history_encoder(history)
                lengths = history_mask.long().sum(dim=1).clamp(min=1) - 1
                history_value = output[torch.arange(output.shape[0], device=output.device), lengths]
            else:
                projected = self.history_projection(history) + self.position_embedding[:, : history.shape[1]]
                output = self.history_encoder(projected, src_key_padding_mask=~history_mask.bool())
                weights = history_mask.float().unsqueeze(-1)
                history_value = (output * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
            fused = self.fusion(torch.cat((static_value, history_value), dim=-1))
            all_scores = torch.cat([head(fused) for head in self.role_heads], dim=1)
            score = all_scores.gather(1, role_index.long().view(-1, 1)).squeeze(1)
            return score, self.opponent_head(fused).view(-1, 2, 16), self.teammate_head(fused).squeeze(1)

    return ActionRanker()


def build_teacher(width: int = 384) -> Any:
    torch, nn = require_torch()

    class PrivilegedTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(STATIC_FEATURE_SIZE + TEACHER_FEATURE_SIZE, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, 3),
            )

        def forward(self, static, hidden, role_index):
            scores = self.network(torch.cat((static, hidden), dim=-1))
            return scores.gather(1, role_index.long().view(-1, 1)).squeeze(1)

    return PrivilegedTeacher()
