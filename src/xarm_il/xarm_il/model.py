"""Compact action-chunk Transformer policy for multi-camera xArm observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    camera_count: int
    state_dim: int
    action_dim: int = 3
    horizon: int = 20
    d_model: int = 256
    nhead: int = 8
    decoder_layers: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ActionChunkTransformer(nn.Module):
    """Predict a fixed Cartesian velocity sequence from images and robot state."""

    def __init__(self, config: ModelConfig, pretrained_backbone: bool = False) -> None:
        super().__init__()
        if config.d_model % config.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.config = config
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:  # pragma: no cover - installation error path
            raise RuntimeError("torchvision is required for the policy model") from exc

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.camera_projection = nn.Linear(feature_dim, config.d_model)
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.camera_embedding = nn.Embedding(config.camera_count, config.d_model)
        self.action_query = nn.Embedding(config.horizon, config.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.action_dim),
        )

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("images must have shape [batch, camera, channel, height, width]")
        batch_size, camera_count, channels, height, width = images.shape
        if camera_count != self.config.camera_count:
            raise ValueError(
                f"Expected {self.config.camera_count} cameras, received {camera_count}"
            )
        flat_images = images.reshape(batch_size * camera_count, channels, height, width)
        camera_features = self.backbone(flat_images)
        camera_tokens = self.camera_projection(camera_features).reshape(
            batch_size, camera_count, self.config.d_model
        )
        camera_ids = torch.arange(camera_count, device=images.device)
        camera_tokens = camera_tokens + self.camera_embedding(camera_ids).unsqueeze(0)
        state_token = self.state_encoder(state).unsqueeze(1)
        memory = torch.cat([state_token, camera_tokens], dim=1)
        queries = self.action_query.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.output_head(decoded)


def masked_action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    smoothness_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Smooth-L1 imitation loss with optional adjacent-action regularization."""
    element_loss = nn.functional.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
    valid = mask.to(element_loss.dtype)
    imitation = (element_loss * valid).sum() / valid.sum().clamp_min(1.0)

    if prediction.shape[1] > 1:
        adjacent_valid = (mask[:, 1:] & mask[:, :-1]).to(prediction.dtype)
        delta = (prediction[:, 1:] - prediction[:, :-1]).square().mean(-1)
        smoothness = (delta * adjacent_valid).sum() / adjacent_valid.sum().clamp_min(1.0)
    else:
        smoothness = prediction.new_zeros(())
    total = imitation + float(smoothness_weight) * smoothness
    return total, {"imitation": imitation.detach(), "smoothness": smoothness.detach()}
