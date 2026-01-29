"""
Model C: Change Point Detection for Anomaly Onset Identification

This model detects WHEN an anomaly starts in a sequence of cockpit voice recordings,
rather than just classifying each utterance. It uses distribution shift detection
between consecutive sliding windows.

Architecture:
1. BERT encoder (frozen or fine-tuned) for utterance embeddings
2. Sliding window dissimilarity computation
3. Change point detection via peak detection or learned classifier
4. Evaluation: MAE in time/utterance prediction

Research Contribution:
- First work to pinpoint anomaly onset in CVR analysis
- Multiple distribution shift metrics compared
- Temporal smoothness constraints
"""
from typing import Dict, List, Optional, Tuple, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np


class UtteranceEmbedder(nn.Module):
    """BERT-based utterance embedder."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        freeze: bool = True,
        pooling: str = "cls",  # cls, mean, max
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.pooling = pooling
        
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len, max_token_len)
            attention_mask: (batch, seq_len, max_token_len)
        Returns:
            embeddings: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, max_token_len = input_ids.shape
        
        # Flatten for BERT
        input_ids_flat = input_ids.view(batch_size * seq_len, max_token_len)
        attention_mask_flat = attention_mask.view(batch_size * seq_len, max_token_len)
        
        # Encode
        outputs = self.bert(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        hidden_states = outputs.last_hidden_state  # (batch*seq_len, max_token_len, hidden)
        
        # Pool
        if self.pooling == "cls":
            embeddings = hidden_states[:, 0, :]  # [CLS] token
        elif self.pooling == "mean":
            mask_expanded = attention_mask_flat.unsqueeze(-1).float()
            embeddings = (hidden_states * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        elif self.pooling == "max":
            embeddings = hidden_states.max(1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Reshape back
        embeddings = embeddings.view(batch_size, seq_len, self.hidden_size)
        return embeddings


class DistributionShiftMetrics:
    """Compute distribution shift between consecutive windows."""
    
    @staticmethod
    def cosine_dissimilarity(
        window1: torch.Tensor,
        window2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cosine dissimilarity between two windows.
        
        Args:
            window1: (batch, window_len, hidden)
            window2: (batch, window_len, hidden)
        Returns:
            dissimilarity: (batch,) scalar per batch
        """
        # Mean pool over window
        if mask1 is not None:
            mask1 = mask1.unsqueeze(-1).float()
            vec1 = (window1 * mask1).sum(1) / mask1.sum(1).clamp(min=1)
        else:
            vec1 = window1.mean(1)
            
        if mask2 is not None:
            mask2 = mask2.unsqueeze(-1).float()
            vec2 = (window2 * mask2).sum(1) / mask2.sum(1).clamp(min=1)
        else:
            vec2 = window2.mean(1)
        
        # Cosine similarity -> dissimilarity
        similarity = F.cosine_similarity(vec1, vec2, dim=1)
        dissimilarity = 1 - similarity
        return dissimilarity
    
    @staticmethod
    def mmd_rbf(
        window1: torch.Tensor,
        window2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Maximum Mean Discrepancy with RBF kernel.
        More sensitive to distribution differences than cosine.
        """
        # Filter by mask if provided
        if mask1 is not None:
            window1 = window1[mask1.bool()]
        if mask2 is not None:
            window2 = window2[mask2.bool()]
        
        # Reshape to (n_samples, hidden)
        if window1.dim() == 3:
            batch_size = window1.size(0)
            window1 = window1.view(-1, window1.size(-1))
            window2 = window2.view(-1, window2.size(-1))
        else:
            batch_size = 1
            window1 = window1.unsqueeze(0) if window1.dim() == 1 else window1
            window2 = window2.unsqueeze(0) if window2.dim() == 1 else window2
        
        # MMD computation
        xx = torch.cdist(window1, window1).pow(2)
        yy = torch.cdist(window2, window2).pow(2)
        xy = torch.cdist(window1, window2).pow(2)
        
        gamma = 1 / (2 * sigma ** 2)
        xx_kernel = torch.exp(-gamma * xx)
        yy_kernel = torch.exp(-gamma * yy)
        xy_kernel = torch.exp(-gamma * xy)
        
        mmd = xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()
        return mmd


class LearnableChangePointDetector(nn.Module):
    """Learnable head for change point detection from dissimilarity curve."""
    
    def __init__(
        self,
        input_dim: int = 1,  # Dissimilarity score
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Temporal encoder for dissimilarity sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, hidden_dim))
        
        # Output heads
        self.change_point_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Confidence head (uncertainty estimation)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        dissimilarity_sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            dissimilarity_sequence: (batch, seq_len, 1) dissimilarity scores
            mask: (batch, seq_len) valid positions
        Returns:
            Dict with change_point (batch,), confidence (batch,), logits (batch, seq_len)
        """
        batch_size, seq_len, _ = dissimilarity_sequence.shape
        
        # Project input
        x = self.input_proj(dissimilarity_sequence)  # (batch, seq_len, hidden)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Create key padding mask
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()
        
        # Temporal encoding
        x = self.temporal_encoder(x, src_key_padding_mask=key_padding_mask)
        
        # Per-position change point probability
        logits = self.change_point_head(x).squeeze(-1)  # (batch, seq_len)
        
        # Softmax over valid positions
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float('-inf'))
        
        probs = F.softmax(logits, dim=1)
        
        # Expected value as change point
        positions = torch.arange(seq_len, device=logits.device).float()
        change_point = (probs * positions.unsqueeze(0)).sum(1)  # (batch,)
        
        # Confidence (max probability)
        confidence = probs.max(1)[0]
        
        return {
            "change_point": change_point,
            "confidence": confidence,
            "logits": logits,
            "probabilities": probs,
        }


class ChangePointDetector(nn.Module):
    """
    Complete Change Point Detection model.
    
    Detects the transition point from normal to anomalous communication.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        freeze_encoder: bool = True,
        window_size: int = 5,
        shift_metric: str = "cosine",
        smoothing_window: int = 3,
        use_learnable_detector: bool = True,
        detector_hidden: int = 256,
        detector_layers: int = 2,
        max_utterances: int = 50,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.window_size = window_size
        self.shift_metric = shift_metric
        self.smoothing_window = smoothing_window
        self.max_utterances = max_utterances
        
        # Utterance encoder
        self.embedder = UtteranceEmbedder(model_name, freeze=freeze_encoder)
        
        # Change point detector
        if use_learnable_detector:
            self.detector = LearnableChangePointDetector(
                input_dim=1,
                hidden_dim=detector_hidden,
                num_layers=detector_layers,
                dropout=dropout,
                max_seq_len=max_utterances,
            )
        else:
            self.detector = None
        
        # Learnable smoothing (optional)
        if smoothing_window > 1:
            self.smooth_conv = nn.Conv1d(1, 1, smoothing_window, padding=smoothing_window//2)
        else:
            self.smooth_conv = None
    
    def compute_dissimilarity_curve(
        self,
        embeddings: torch.Tensor,
        utterance_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dissimilarity curve using sliding windows.
        
        Args:
            embeddings: (batch, seq_len, hidden)
            utterance_mask: (batch, seq_len) - 1 for valid, 0 for padding
        Returns:
            dissimilarity: (batch, num_windows) dissimilarity scores
        """
        batch_size, seq_len, hidden = embeddings.shape
        device = embeddings.device
        
        # Number of windows
        num_windows = seq_len - self.window_size + 1
        if num_windows <= 0:
            # Sequence too short, return zeros
            return torch.zeros(batch_size, 1, device=device)
        
        dissimilarities = []
        
        for i in range(num_windows):
            # Current window
            window1 = embeddings[:, i:i+self.window_size, :]
            mask1 = utterance_mask[:, i:i+self.window_size]
            
            # Next window
            if i + self.window_size < seq_len:
                window2 = embeddings[:, i+self.window_size:i+2*self.window_size, :]
                mask2 = utterance_mask[:, i+self.window_size:i+2*self.window_size]
            else:
                # Pad with last window
                window2 = window1
                mask2 = mask1
            
            # Compute dissimilarity
            if self.shift_metric == "cosine":
                dissim = DistributionShiftMetrics.cosine_dissimilarity(
                    window1, window2, mask1, mask2
                )
            elif self.shift_metric == "mmd":
                dissim = DistributionShiftMetrics.mmd_rbf(
                    window1, window2, mask1, mask2
                )
            else:
                raise ValueError(f"Unknown shift metric: {self.shift_metric}")
            
            dissimilarities.append(dissim)
        
        dissimilarity_curve = torch.stack(dissimilarities, dim=1)  # (batch, num_windows)
        
        # Smoothing
        if self.smooth_conv is not None:
            diss_smooth = self.smooth_conv(dissimilarity_curve.unsqueeze(1)).squeeze(1)
            # Ensure non-negative
            diss_smooth = F.relu(diss_smooth)
            return diss_smooth
        
        return dissimilarity_curve
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        utterance_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, max_utterances, max_length)
            attention_mask: (batch, max_utterances, max_length)
            utterance_mask: (batch, max_utterances) - which utterances are valid
        
        Returns:
            Dict with change_point predictions and intermediate values
        """
        batch_size, max_utterances, max_length = input_ids.shape
        device = input_ids.device
        
        # Create default mask if not provided
        if utterance_mask is None:
            utterance_mask = torch.ones(batch_size, max_utterances, device=device)
        
        # Encode utterances
        embeddings = self.embedder(input_ids, attention_mask)
        
        # Compute dissimilarity curve
        dissimilarity_curve = self.compute_dissimilarity_curve(embeddings, utterance_mask)
        
        # Detect change point
        if self.detector is not None:
            # Use learnable detector
            diss_input = dissimilarity_curve.unsqueeze(-1)  # (batch, num_windows, 1)
            
            # Create mask for valid windows
            num_windows = diss_input.size(1)
            valid_windows = torch.ones(batch_size, num_windows, device=device)
            
            result = self.detector(diss_input, valid_windows)
            result["dissimilarity_curve"] = dissimilarity_curve
            result["embeddings"] = embeddings
            return result
        else:
            # Use heuristic: argmax of dissimilarity
            change_point = dissimilarity_curve.argmax(1).float()
            confidence = dissimilarity_curve.max(1)[0]
            
            return {
                "change_point": change_point,
                "confidence": confidence,
                "dissimilarity_curve": dissimilarity_curve,
                "embeddings": embeddings,
            }


def create_model(**kwargs) -> ChangePointDetector:
    """Factory function to create ChangePointDetector."""
    return ChangePointDetector(**kwargs)


def create_tokenizer(model_name: str = "bert-base-uncased"):
    """Factory function to create tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    # Test the model
    print("Testing Change Point Detection model...")
    
    model = create_model(
        window_size=5,
        shift_metric="cosine",
        use_learnable_detector=True,
    )
    tokenizer = create_tokenizer()
    
    # Create dummy data
    batch_size = 2
    max_utterances = 20
    max_length = 128
    
    input_ids = torch.randint(0, 30522, (batch_size, max_utterances, max_length))
    attention_mask = torch.ones(batch_size, max_utterances, max_length)
    utterance_mask = torch.ones(batch_size, max_utterances)
    
    print("Running forward pass...")
    output = model(input_ids, attention_mask, utterance_mask)
    
    print(f"Change point: {output['change_point']}")
    print(f"Confidence: {output['confidence']}")
    print(f"Dissimilarity curve shape: {output['dissimilarity_curve'].shape}")
