"""
Utilities for visualizing QFormer attention weights as heatmaps over images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, Dict


def extract_qformer_attention_weights(
    qformer: nn.Module,
    visual_feats: torch.Tensor,  # (B, num_visual_tokens, D)
    visual_attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Extract attention weights from QFormer by manually computing cross-attention.

    Args:
        qformer: QFormerVisualSampler instance
        visual_feats: (B, num_visual_tokens, D) - visual tokens
        visual_attn_mask: Optional (B, num_visual_tokens) - padding mask

    Returns:
        attention_weights: (B, num_query, num_visual_tokens) - attention weights
    """
    B, num_visual_tokens, D = visual_feats.shape
    num_query = qformer.num_query

    # Get query tokens
    query_tokens = qformer.query_tokens.unsqueeze(0).expand(
        B, -1, -1
    )  # (B, num_query, D)
    query_tokens = query_tokens + qformer.query_pos.unsqueeze(0)

    # Project visual features
    visual_feats_proj = qformer.visual_proj(visual_feats)  # (B, num_visual_tokens, D)

    # Extract the first decoder layer's attention
    decoder_layer = qformer.decoder.layers[0]

    # Get Q, K, V projections from the decoder layer
    # The decoder layer uses MultiheadAttention for cross-attention
    attn_module = decoder_layer.multihead_attn

    # MultiheadAttention uses a single in_proj_weight containing Q, K, V concatenated
    # Shape: (3 * embed_dim, embed_dim)
    embed_dim = attn_module.embed_dim

    # Ensure dimensions match (visual_proj should output embed_dim)
    if D != embed_dim:
        # If dimensions don't match, we need to project first
        # This shouldn't happen if QFormer is set up correctly, but handle it gracefully
        raise ValueError(
            f"Dimension mismatch: visual_feats has dimension {D} but attention module expects {embed_dim}. "
            "This suggests the visual_proj in QFormer is not configured correctly."
        )

    # Split the weight matrix into Q, K, V projections
    if attn_module.in_proj_weight is not None:
        q_proj_weight, k_proj_weight, v_proj_weight = attn_module.in_proj_weight.chunk(
            3, dim=0
        )
    else:
        # If using separate q_proj_weight, k_proj_weight, v_proj_weight
        q_proj_weight = attn_module.q_proj_weight
        k_proj_weight = attn_module.k_proj_weight
        v_proj_weight = attn_module.v_proj_weight

    # Split bias if it exists
    if attn_module.in_proj_bias is not None:
        q_bias, k_bias, v_bias = attn_module.in_proj_bias.chunk(3, dim=0)
    else:
        q_bias = k_bias = v_bias = None

    # Project queries, keys, values manually
    q = F.linear(query_tokens, q_proj_weight, q_bias)  # (B, num_query, embed_dim)
    k = F.linear(
        visual_feats_proj, k_proj_weight, k_bias
    )  # (B, num_visual_tokens, embed_dim)
    v = F.linear(
        visual_feats_proj, v_proj_weight, v_bias
    )  # (B, num_visual_tokens, embed_dim)

    # Reshape for multi-head attention
    num_heads = attn_module.num_heads
    head_dim = attn_module.embed_dim // num_heads

    q = q.reshape(B, num_query, num_heads, head_dim).transpose(
        1, 2
    )  # (B, num_heads, num_query, head_dim)
    k = k.reshape(B, num_visual_tokens, num_heads, head_dim).transpose(
        1, 2
    )  # (B, num_heads, num_visual_tokens, head_dim)

    # Compute attention scores
    scale = head_dim**-0.5
    attn_scores = (
        torch.matmul(q, k.transpose(-2, -1)) * scale
    )  # (B, num_heads, num_query, num_visual_tokens)

    # Apply mask if provided
    if visual_attn_mask is not None:
        # Convert to attention mask format (True = mask out)
        attn_mask = visual_attn_mask.unsqueeze(1).unsqueeze(
            2
        )  # (B, 1, 1, num_visual_tokens)
        attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

    # Softmax to get attention weights
    attn_weights = F.softmax(
        attn_scores, dim=-1
    )  # (B, num_heads, num_query, num_visual_tokens)

    # Average across heads
    attn_weights = attn_weights.mean(dim=1)  # (B, num_query, num_visual_tokens)

    return attn_weights


def create_attention_heatmap(
    attention_weights: torch.Tensor,  # (num_query, num_visual_tokens) or (B, num_query, num_visual_tokens)
    image_shape: Tuple[int, int],  # (H, W) of original image
    patch_size: int = 16,  # Size of each patch (e.g., 16x16 for ViT)
    num_cameras: int = 1,  # Number of cameras (visual tokens are concatenated)
    camera_idx: int = 0,  # Which camera to visualize (0-indexed)
) -> np.ndarray:
    """
    Create a heatmap from attention weights by mapping them to image spatial locations.

    Args:
        attention_weights: (num_query, num_visual_tokens) - attention weights for each query
        image_shape: (H, W) - original image dimensions
        patch_size: Size of each patch (assumes square patches)
        num_cameras: Number of cameras (tokens are concatenated)
        camera_idx: Which camera to visualize

    Returns:
        heatmap: (H, W) - attention heatmap as numpy array
    """
    if attention_weights.dim() == 3:
        # Take first batch item
        attention_weights = attention_weights[0]

    num_query, num_visual_tokens = attention_weights.shape

    # Calculate tokens per camera
    tokens_per_camera = num_visual_tokens // num_cameras

    # Get tokens for this camera
    start_idx = camera_idx * tokens_per_camera
    end_idx = (camera_idx + 1) * tokens_per_camera
    camera_attn = attention_weights[
        :, start_idx:end_idx
    ]  # (num_query, tokens_per_camera)

    # Average across all queries to get overall attention
    overall_attn = camera_attn.mean(dim=0)  # (tokens_per_camera,)

    # Calculate expected spatial dimensions from image size and patch size
    H, W = image_shape
    expected_patches_h = H // patch_size
    expected_patches_w = W // patch_size
    expected_tokens = expected_patches_h * expected_patches_w

    # Find the best rectangular arrangement for the actual token count
    # Try to match expected dimensions first, otherwise find closest factorization
    if tokens_per_camera == expected_tokens:
        # Perfect match: use expected dimensions
        num_patches_h = expected_patches_h
        num_patches_w = expected_patches_w
    else:
        # Find best rectangular factorization that uses all tokens
        # Collect all valid factorizations
        factorizations = []
        for h in range(1, int(np.sqrt(tokens_per_camera)) + 1):
            if tokens_per_camera % h == 0:
                w = tokens_per_camera // h
                factorizations.append((h, w))

        if factorizations:
            # Prefer factorizations close to expected aspect ratio
            # Also prefer wider arrangements (w > h) for typical image aspect ratios
            best_h, best_w = None, None
            best_score = float("inf")

            for h, w in factorizations:
                # Calculate aspect ratio difference
                if expected_patches_w > 0:
                    aspect_diff = abs(
                        (h / w) - (expected_patches_h / expected_patches_w)
                    )
                else:
                    aspect_diff = float("inf")

                # Prefer wider arrangements (penalize tall arrangements)
                width_bonus = 0.1 if w >= h else 0.3

                score = aspect_diff + width_bonus
                if score < best_score:
                    best_score = score
                    best_h, best_w = h, w

            num_patches_h, num_patches_w = best_h, best_w
        else:
            # Fallback: use closest square (shouldn't happen for valid token counts)
            num_patches_h = int(np.sqrt(tokens_per_camera))
            num_patches_w = (
                tokens_per_camera + num_patches_h - 1
            ) // num_patches_h  # Ceiling division
            # Truncate if needed
            if num_patches_h * num_patches_w > tokens_per_camera:
                overall_attn = overall_attn[: num_patches_h * num_patches_w]

    # Reshape to spatial grid
    attn_grid = (
        overall_attn.reshape(num_patches_h, num_patches_w).detach().cpu().numpy()
    )

    # Upsample to image size
    heatmap = cv2.resize(attn_grid, (W, H), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap


def overlay_heatmap_on_image(
    image: np.ndarray,  # (H, W, 3) - RGB image
    heatmap: np.ndarray,  # (H, W) - attention heatmap
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay attention heatmap on image.

    Args:
        image: (H, W, 3) - RGB image
        heatmap: (H, W) - attention heatmap [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap (e.g., cv2.COLORMAP_JET)

    Returns:
        overlaid: (H, W, 3) - image with heatmap overlay
    """
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlaid = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)

    return overlaid


def visualize_qformer_attention(
    qformer: nn.Module,
    visual_feats: torch.Tensor,  # (B, num_visual_tokens, D)
    image: np.ndarray,  # (H, W, 3) - RGB image
    visual_attn_mask: Optional[torch.Tensor] = None,
    patch_size: int = 16,
    num_cameras: int = 1,
    camera_idx: int = 0,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Complete pipeline: extract attention weights and create visualization.

    Returns:
        overlaid_image: (H, W, 3) - image with attention heatmap overlay
        attention_weights: (B, num_query, num_visual_tokens) - raw attention weights
    """
    # Extract attention weights
    attention_weights = extract_qformer_attention_weights(
        qformer, visual_feats, visual_attn_mask
    )

    # Create heatmap
    H, W = image.shape[:2]
    heatmap = create_attention_heatmap(
        attention_weights,
        image_shape=(H, W),
        patch_size=patch_size,
        num_cameras=num_cameras,
        camera_idx=camera_idx,
    )

    # Overlay on image
    overlaid = overlay_heatmap_on_image(image, heatmap, alpha=alpha)

    return overlaid, attention_weights
