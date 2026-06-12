import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128, gated: bool = False):
        super().__init__()
        self.gated = gated
        if gated:
            self.attn_a = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh())
            self.attn_b = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid())
            self.attn_c = nn.Linear(hidden_dim, 1)
        else:
            self.attention = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, feats):
        if self.gated:
            logits = self.attn_c(self.attn_a(feats) * self.attn_b(feats)).squeeze(1)
        else:
            logits = self.attention(feats).squeeze(1)
        weights = F.softmax(logits, dim=0)
        bag = torch.sum(weights.unsqueeze(1) * feats, dim=0)
        return bag, logits


class MILClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int, gated_attention: bool = False):
        super().__init__()
        self.mil = AttentionMIL(dim=dim, hidden_dim=128, gated=gated_attention)
        self.cls = nn.Linear(dim, num_classes)

    def forward(self, feats):
        bag, attn_logits = self.mil(feats)
        logits = self.cls(bag)
        return logits, bag, attn_logits


class StudentMIL(MILClassifier):
    def __init__(self, dim: int, num_classes: int, gated_attention: bool = False):
        super().__init__(dim=dim, num_classes=num_classes, gated_attention=gated_attention)
        self.inst_classifier = nn.Linear(dim, 2)


class StudentBackbone(nn.Module):
    def __init__(self, conch_model: nn.Module, layers_to_return: Iterable[int] = (0, 1, 10)):
        super().__init__()
        self.conch_model = conch_model
        self.visual = conch_model.visual.trunk
        self.layers_to_return = sorted(list(layers_to_return))
        self.embed_dim = 768

    def forward(self, x: torch.Tensor):
        trunk = self.visual
        batch_size = x.shape[0]
        x = trunk.patch_embed(x)
        if x.dim() == 4:
            batch_size, _, _, _ = x.shape
            x = x.flatten(2).transpose(1, 2)
        grid_size = int(math.sqrt(x.shape[1]))

        cls_token = trunk.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + trunk.pos_embed[:, : x.shape[1], :]
        x = trunk.pos_drop(x)

        layer_cls = {}
        layer_tokens = {}
        for idx, block in enumerate(trunk.blocks):
            x = block(x)
            if idx in self.layers_to_return:
                layer_cls[idx] = x[:, 0, :]
                layer_tokens[idx] = x[:, 1:, :].view(batch_size, grid_size, grid_size, -1)

        x = trunk.norm(x)
        return {
            "final": x[:, 0, :],
            "layer_cls": layer_cls,
            "layer_tokens": layer_tokens,
        }
