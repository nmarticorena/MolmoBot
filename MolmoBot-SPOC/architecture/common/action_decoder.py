from typing import Optional, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from molmobot_spoc.architecture.common.utils import (
    PositionalEncoder,
)

from molmobot_spoc.architecture.common.goal_cond_llama import (
    TransformerDecoder as GoalCondLlamaDecoder,
)
from molmobot_spoc.architecture.config.model_config import ActionDecoderConfig


class QFormerVisualSampler(nn.Module):
    """
    QFormer-based sampler for visual tokens.

    Uses learnable query tokens to cross-attend to both text and visual features,
    sampling a fixed number of tokens from potentially many visual tokens.
    This is similar to BLIP-2's QFormer approach.

    Args:
        num_query: Number of query tokens to sample (e.g., 32)
        hidden_size: Hidden dimension size (should match visual token dim)
        num_layers: Number of transformer decoder layers
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        num_query: int = 32,
        hidden_size: int = 768,
        num_layers: int = 3,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_query = num_query
        self.hidden_size = hidden_size

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_query, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        # Learnable query positional embeddings
        self.query_pos = nn.Parameter(torch.randn(num_query, hidden_size))
        nn.init.normal_(self.query_pos, std=0.02)

        # Transformer decoder for cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=hidden_size * 4,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Optional projection for visual features (if needed for dimension matching)
        self.visual_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        visual_feats: torch.Tensor,  # (B, num_visual_tokens, hidden_size)
        text_feats: Optional[
            torch.Tensor
        ] = None,  # (B, num_text_tokens, hidden_size) - text features to cross-attend to
    ) -> torch.Tensor:
        """
        Sample visual tokens using QFormer queries.

        Args:
            visual_feats: (B, num_visual_tokens, hidden_size) - visual tokens to sample from
            text_feats: Optional (B, num_text_tokens, hidden_size) - text features to also cross-attend to

        Returns:
            sampled_tokens: (B, num_query, hidden_size) - sampled visual tokens
        """
        B = visual_feats.size(0)

        # Project visual features if needed
        visual_feats = self.visual_proj(visual_feats)

        # Expand query tokens for batch: (num_query, hidden_size) -> (B, num_query, hidden_size)
        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        query_tokens = query_tokens + self.query_pos.unsqueeze(0)

        # Build memory from visual and text features
        if text_feats is not None:
            # Concatenate text and visual features: (B, num_text_tokens + num_visual_tokens, hidden_size)
            memory = torch.cat([text_feats, visual_feats], dim=1)

        else:
            # Only visual features
            memory = visual_feats

        memory_key_padding_mask = None

        # Cross-attention: queries attend to memory (text + visual features)
        sampled_tokens = self.decoder(
            tgt=query_tokens,  # (B, num_query, hidden_size) - queries
            memory=memory,  # (B, num_memory_tokens, hidden_size) - keys/values (text + visual)
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Store visual features and query tokens for attention extraction
        self._last_visual_feats = visual_feats
        self._last_query_tokens = query_tokens
        self._last_memory_mask = memory_key_padding_mask

        return sampled_tokens  # (B, num_query, hidden_size)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Extract attention weights from the last forward pass.

        Returns:
            attention_weights: (B, num_query, num_visual_tokens) or None if not available
        """
        if not hasattr(self, "_last_visual_feats") or self._last_visual_feats is None:
            return None

        from molmobot_spoc.utils.qformer_attention_viz import (
            extract_qformer_attention_weights,
        )

        return extract_qformer_attention_weights(
            self,
            self._last_visual_feats,
            self._last_memory_mask,
        )


class ParallelActionDecoder(nn.Module):
    """
    Action decoder with learnable action queries - NO HISTORY version.

    Uses only the CURRENT timestep for visual and proprioception information.
    Optionally accepts full spatial visual tokens for better spatial understanding.

    Key features:
    1. Uses LEARNABLE query embeddings (not derived from observations)
    2. Visual/proprioception accessed ONLY through cross-attention
    3. No previous actions in the decoder input
    4. Bidirectional self-attention among action queries
    5. Uses LAST timestep only (no temporal history)
    6. Optional QFormer sampling: can sample a fixed number of visual tokens
       instead of passing all tokens (prevents attention dilution)
    """

    def __init__(
        self,
        cfg: ActionDecoderConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.decoder = GoalCondLlamaDecoder(cfg.decoder)

        # Get parameters from action_space and decoder config
        action_dim = cfg.action_space.action_dim
        chunk_size = cfg.action_space.chunk_size
        dim = cfg.decoder.dim

        # Fixed sequence length = action_dim * chunk_size
        self.fixed_seq_len = action_dim * chunk_size

        # ============ LEARNABLE ACTION QUERIES ============
        self.action_queries = nn.Parameter(torch.zeros(self.fixed_seq_len, dim))
        nn.init.normal_(self.action_queries, std=0.02)

        # Positional encoder for action query positions
        self.position_encoder = PositionalEncoder(dim, self.fixed_seq_len)

        # ============ CROSS-ATTENTION MEMORY COMPONENTS ============
        if cfg.use_proprioception:
            self.proprioception_proj = nn.Linear(
                action_dim + cfg.proprio_additional_dim, dim
            )
            nn.init.uniform_(self.proprioception_proj.weight, -0.01, 0.01)
            nn.init.zeros_(self.proprioception_proj.bias)

        # QFormer visual token sampler (optional)
        if cfg.use_qformer_sampling:
            self.qformer = QFormerVisualSampler(
                num_query=cfg.qformer_num_query,
                hidden_size=dim,
                num_layers=cfg.qformer_num_layers,
                num_heads=cfg.qformer_num_heads,
            )
        else:
            self.qformer = None

    def get_query_embeddings(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Create learned query embeddings with positional encoding."""
        query_embed = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)

        pos_ids = torch.arange(self.fixed_seq_len, device=device)
        pos_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)
        pos_enc = self.position_encoder(pos_ids)

        return query_embed + pos_enc

    def build_cross_attention_memory(
        self,
        goals_features: torch.Tensor,  # (B, L, D)
        proprioception: Optional[torch.Tensor] = None,
        visual_tokens: Optional[torch.Tensor] = None,  # (B, T, num_visual_tokens, D)
        point_tokens: Optional[
            torch.Tensor
        ] = None,  # (B, T, N) raw pixel coords in [0, 1]
    ) -> torch.Tensor:
        """
        Build cross-attention memory using LAST timestep only (no history).

        Memory structure:
        - Text goal features (L tokens)
        - Visual tokens from LAST timestep:
            * If QFormer enabled: sampled to num_query tokens (Q-Former cross-attends to text)
            * If QFormer disabled: all num_visual_tokens passed directly
        - Point tokens from LAST timestep (N tokens, embedded here like proprioception)
        - Proprioception from LAST timestep (1 token)
        """
        memory_components = [goals_features]

        # Visual: LAST timestep only
        visual_tokens_last = visual_tokens[:, -1, :, :]  # (B, num_visual_tokens, D)
        if self.qformer is not None:
            sampled_visual = self.qformer(
                visual_feats=visual_tokens_last,
                text_feats=goals_features,
            )  # (B, num_query, D)
            memory_components.append(sampled_visual)
        else:
            memory_components.append(visual_tokens_last)

        # Point embeddings: LAST timestep only.
        # point_tokens arrives as (B*T, N, D) from adapt_goal_points_features.
        if point_tokens is not None:
            B = goals_features.shape[0]
            T = point_tokens.shape[0] // B
            N = point_tokens.shape[1]
            point_tokens_last = point_tokens.reshape(B, T, N, -1)[:, -1, :]  # (B, N, D)
            memory_components.append(point_tokens_last)

        # Proprioception: LAST timestep only
        if self.cfg.use_proprioception and proprioception is not None:
            proprio_last = proprioception[:, -1:, :]
            proprio_tokens = self.proprioception_proj(proprio_last)
            memory_components.append(proprio_tokens)

        return torch.cat(memory_components, dim=1)

    def forward(
        self,
        goals_features: torch.Tensor,  # B, L, D
        padding_mask: Optional[torch.Tensor] = None,
        proprioception: Optional[torch.Tensor] = None,
        visual_tokens: Optional[torch.Tensor] = None,  # (B, T, num_visual_tokens, D)
        point_tokens: Optional[
            torch.Tensor
        ] = None,  # (B, T, N) raw pixel coords in [0, 1]
    ):
        """
        Forward pass using LAST timestep only (no history).

        Args:
            visual_tokens: If provided, uses ALL spatial tokens from current frame
            point_coords: If provided, raw pixel coords embedded inside the decoder
        """
        _ref = visual_tokens if visual_tokens is not None else goals_features
        B = _ref.shape[0]
        device = _ref.device

        query_embed = self.get_query_embeddings(B, device)

        memory = self.build_cross_attention_memory(
            goals_features=goals_features,
            proprioception=proprioception,
            visual_tokens=visual_tokens,
            point_tokens=point_tokens,
        )

        decoder_output = self.decoder(
            tokens=query_embed,
            memory=memory,
            start_pos=0,
            mask=torch.zeros((self.fixed_seq_len, self.fixed_seq_len), device=device),
            padding_mask=padding_mask,
        )

        return decoder_output
