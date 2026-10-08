import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights, ViT_B_16_Weights
from typing import Tuple, Dict, Any, Optional


class TokenTextEncoder(nn.Module):
    """
    Token-Level Natural Language Representation Encoder.
    Preserves sequence tokens [B, seq_len, D] for token-to-patch cross-modal interaction.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def expand_vocab(self, new_vocab_size: int):
        if new_vocab_size <= self.embedding.num_embeddings:
            return
        old_embeddings = self.embedding.weight.data
        new_embedding = nn.Embedding(new_vocab_size, self.embed_dim, padding_idx=0)
        nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            new_embedding.weight[:old_embeddings.shape[0]] = old_embeddings
        self.embedding = new_embedding

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens: [B, seq_len, embed_dim]
            pooled: [B, embed_dim]
            mask:   [B, seq_len]
        """
        mask = (x != 0)  # [B, seq_len]
        emb = self.dropout(self.embedding(x))
        h, _ = self.gru(emb)  # [B, seq_len, hidden_dim * 2]
        tokens = self.layer_norm(self.proj(h))  # [B, seq_len, embed_dim]

        # Length-normalized masked mean pooling
        mask_expanded = mask.unsqueeze(-1).float()
        sum_tokens = torch.sum(tokens * mask_expanded, dim=1)
        lengths = torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        pooled = sum_tokens / lengths

        return tokens, pooled, mask


class PatchVisionEncoder(nn.Module):
    """
    Patch-Level Visual Representation Encoder.
    Extracts spatial grid patch tokens [B, num_patches, D] (e.g. 7x7=49 visual patches)
    and a global image representation [B, D] for token-to-patch bidirectional cross-attention.
    """
    def __init__(self, out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            mobilenet = models.mobilenet_v3_small(weights=weights)
        except Exception:
            mobilenet = models.mobilenet_v3_small(weights=None)

        self.features = mobilenet.features  # output: [B, 576, 7, 7]
        self.patch_proj = nn.Sequential(
            nn.Conv2d(576, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            patches: [B, num_patches, out_dim] (e.g., 49 spatial patches)
            pooled:  [B, out_dim]
        """
        feat = self.features(x)  # [B, 576, H, W]
        proj_feat = self.patch_proj(feat)  # [B, out_dim, H, W]
        B, C, H, W = proj_feat.shape

        # Reshape to spatial patch sequence [B, H*W, C]
        patches = proj_feat.flatten(2).transpose(1, 2)  # [B, num_patches, out_dim]
        patches = self.layer_norm(patches)

        pooled = self.global_pool(proj_feat).view(B, C)
        pooled = self.global_proj(pooled)

        return patches, pooled


class BidirectionalCrossModalAttention(nn.Module):
    """
    Bidirectional Cross-Modal Attention Engine.
    1. Text -> Image: Allows text tokens to query and attend to relevant visual patches.
    2. Image -> Text: Allows visual patches to query and attend to relevant text tokens.
    Stores cross-attention maps for explainability visualizations and faithfulness testing.
    """
    def __init__(self, feature_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Text -> Image Multi-Head Attention
        self.t2i_mha = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.t_norm = nn.LayerNorm(feature_dim)

        # Image -> Text Multi-Head Attention
        self.i2t_mha = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.i_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        image_patches: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_attended:  [B, text_dim]
            image_attended: [B, img_dim]
            t2i_attn_map:   [B, num_text_tokens, num_img_patches]
            i2t_attn_map:   [B, num_img_patches, num_text_tokens]
        """
        # Key padding mask for text: True where padded
        key_padding_mask = (~text_mask) if text_mask is not None else None

        # 1. Text attends to Image (Q=Text, K=Image, V=Image)
        t_out, t2i_attn_weights = self.t2i_mha(
            query=text_tokens,
            key=image_patches,
            value=image_patches
        )
        t_enhanced = self.t_norm(text_tokens + t_out)
        text_attended = t_enhanced.mean(dim=1)  # [B, D]

        # 2. Image attends to Text (Q=Image, K=Text, V=Text)
        i_out, i2t_attn_weights = self.i2t_mha(
            query=image_patches,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=key_padding_mask
        )
        i_enhanced = self.i_norm(image_patches + i_out)
        image_attended = i_enhanced.mean(dim=1)  # [B, D]

        return text_attended, image_attended, t2i_attn_weights, i2t_attn_weights


class ConflictAwarenessHead(nn.Module):
    """
    Core Research Component: Explicit Text-Image Conflict / Incongruity Detector.
    Evaluates agreement vs. discordance between modalities:
    - Text Positive + Image Negative -> High Conflict
    - Text Negative + Image Positive -> High Conflict (Sarcasm / Irony)
    - Text Positive + Image Positive -> Low Conflict (Strong Agreement)
    Outputs: conflict probability in [0, 1].
    """
    def __init__(self, feature_dim: int = 128, num_classes: int = 3):
        super().__init__()
        # Cross-modality disparity interaction: [t, i, |t-i|, t*i]
        self.conflict_mlp = nn.Sequential(
            nn.Linear(feature_dim * 4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Independent modality sentiment classifiers
        self.text_sentiment_head = nn.Linear(feature_dim, num_classes)
        self.image_sentiment_head = nn.Linear(feature_dim, num_classes)

    def forward(self, text_rep: torch.Tensor, image_rep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            conflict_prob: [B, 1]
            text_logits:   [B, num_classes]
            image_logits:  [B, num_classes]
        """
        diff = torch.abs(text_rep - image_rep)
        prod = text_rep * image_rep
        interaction = torch.cat([text_rep, image_rep, diff, prod], dim=-1)

        conflict_prob = self.conflict_mlp(interaction)  # [B, 1]
        text_logits = self.text_sentiment_head(text_rep)
        image_logits = self.image_sentiment_head(image_rep)

        return conflict_prob, text_logits, image_logits


class ConflictGatedFusion(nn.Module):
    """
    Dynamic Gated Fusion with Conflict & Modality-Informativeness Modulation.
    Dynamically assigns trust weights (w_text, w_image) based on:
    1. Text informativeness
    2. Image informativeness
    3. Explicit Conflict Signal
    """
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        # Input: text_rep (D) + img_rep (D) + conflict_prob (1) -> gate weight in [0, 1]
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(
        self,
        text_rep: torch.Tensor,
        image_rep: torch.Tensor,
        conflict_prob: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([text_rep, image_rep, conflict_prob], dim=-1)
        w_text = self.gate_mlp(gate_input)  # [B, 1]
        w_image = 1.0 - w_text               # [B, 1]

        weighted_text = w_text * text_rep
        weighted_image = w_image * image_rep
        fused = self.fusion_proj(torch.cat([weighted_text, weighted_image], dim=-1))

        return fused, w_text, w_image


class ModelOutput(dict):
    """
    Polymorphic Model Output supporting both Dictionary access and Tuple unpacking.
    """
    def __iter__(self):
        return iter([
            self["logits"],
            self["text_weight"],
            self["image_weight"],
            self["text_logits"],
            self["image_logits"]
        ])
    
    def __getitem__(self, item):
        if isinstance(item, int):
            seq = [
                self["logits"],
                self["text_weight"],
                self["image_weight"],
                self["text_logits"],
                self["image_logits"]
            ]
            return seq[item]
        return super().__getitem__(item)


class SentiFusionNet(nn.Module):
    """
    SentiFusion: Attention-Based Multimodal Fusion with Explicit Conflict Awareness.
    Supports:
    - Bidirectional Token-to-Patch Cross-Modal Attention (Text <-> Image)
    - Dedicated Conflict-Awareness Module
    - Conflict-Modulated Dynamic Gated Fusion
    - Modular Ablation Configurations (E1 - E8)
    """
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 3,
        feature_dim: int = 128,
        ablation_mode: str = "full",
        dropout: float = 0.15
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.ablation_mode = ablation_mode

        # 1. Encoders
        self.text_encoder = TokenTextEncoder(vocab_size=vocab_size, embed_dim=feature_dim)
        self.vision_encoder = PatchVisionEncoder(out_dim=feature_dim, pretrained=True)

        # 2. Cross-Modal Attention Engine
        self.cross_attention = BidirectionalCrossModalAttention(feature_dim=feature_dim, num_heads=4)

        # 3. Conflict Awareness Head
        self.conflict_head = ConflictAwarenessHead(feature_dim=feature_dim, num_classes=num_classes)

        # 4. Gated Fusion Unit
        self.gated_fusion = ConflictGatedFusion(feature_dim=feature_dim)

        # 5. Multimodal Sentiment Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

        # Baseline Concatenation Classifier (for E3 baseline)
        self.concat_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_classes)
        )

    def expand_vocab(self, new_vocab_size: int):
        self.text_encoder.expand_vocab(new_vocab_size)

    def forward(
        self,
        text_input: torch.Tensor,
        image_input: torch.Tensor,
        ablation: Optional[str] = None
    ) -> ModelOutput:
        mode = ablation or self.ablation_mode

        # 1. Feature Extraction
        text_tokens, text_pooled, text_mask = self.text_encoder(text_input)
        image_patches, image_pooled = self.vision_encoder(image_input)

        # 2. Cross-Modal Attention
        t_attended, i_attended, t2i_attn, i2t_attn = self.cross_attention(
            text_tokens=text_tokens,
            image_patches=image_patches,
            text_mask=text_mask
        )

        # 3. Conflict Awareness
        conflict_prob, text_logits, image_logits = self.conflict_head(
            text_rep=t_attended,
            image_rep=i_attended
        )

        # 4. Routing based on Ablation Mode
        if mode == "concat":  # E3 Baseline
            concat_features = torch.cat([text_pooled, image_pooled], dim=-1)
            logits = self.concat_classifier(concat_features)
            w_text = torch.full_like(conflict_prob, 0.5)
            w_image = torch.full_like(conflict_prob, 0.5)
            fused = concat_features

        elif mode == "global_features":  # E8 Baseline
            conflict_p, _, _ = self.conflict_head(text_pooled, image_pooled)
            fused, w_text, w_image = self.gated_fusion(text_pooled, image_pooled, conflict_p)
            logits = self.classifier(fused)

        elif mode == "cross_attn_only":  # E4 Baseline
            fused = (t_attended + i_attended) * 0.5
            logits = self.classifier(fused)
            w_text = torch.full_like(conflict_prob, 0.5)
            w_image = torch.full_like(conflict_prob, 0.5)

        elif mode == "unidirectional":  # E7 Baseline (Text -> Image only)
            fused, w_text, w_image = self.gated_fusion(t_attended, image_pooled, conflict_prob)
            logits = self.classifier(fused)

        else:  # E6: Full Proposed Model
            fused, w_text, w_image = self.gated_fusion(t_attended, i_attended, conflict_prob)
            logits = self.classifier(fused)

        return ModelOutput({
            "logits": logits,
            "text_weight": w_text,
            "image_weight": w_image,
            "conflict_prob": conflict_prob,
            "text_logits": text_logits,
            "image_logits": image_logits,
            "t2i_attention": t2i_attn,
            "i2t_attention": i2t_attn,
            "text_tokens": text_tokens,
            "image_patches": image_patches,
            "fused_features": fused
        })


# Aliases for backward compatibility
TextEncoder = TokenTextEncoder
FastVisionEncoder = PatchVisionEncoder
PretrainedVisionEncoder = PatchVisionEncoder
GatedCrossModalAttentionFusion = ConflictGatedFusion
GatedMultimodalFusion = ConflictGatedFusion
