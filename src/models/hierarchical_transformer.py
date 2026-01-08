"""
Model B: Hierarchical Transformer for sequential CVR analysis.

Architecture:
- Per-utterance BERT encoding (token-level)
- Utterance-level Transformer for sequential dependencies
- Multi-head attention over utterances
- Classification head

This model differs from BERT+LSTM by using Transformer layers instead of LSTM
for modeling temporal dependencies between utterances.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for utterance-level sequences."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model))
        nn.init.xavier_uniform_(self.pos_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional_encoding
        """
        seq_len = x.size(1)
        return x + self.pos_embedding[:seq_len, :].unsqueeze(0)


class UtteranceEncoder(nn.Module):
    """Encodes a single utterance using BERT."""

    def __init__(self, model_name: str = "bert-base-uncased", dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # Projection to reduce dimension if needed
        self.projection = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch * max_utterances, max_length)
            attention_mask: (batch * max_utterances, max_length)
        Returns:
            utterance_embeddings: (batch * max_utterances, hidden_size)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_embedding)
        return self.dropout(projected)


class HierarchicalTransformerClassifier(nn.Module):
    """
    Hierarchical Transformer for sequential anomaly detection.

    Architecture:
    1. Token-level: BERT encodes each utterance
    2. Utterance-level: Transformer layers model temporal dependencies
    3. Classification: Final layer predicts anomaly level
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 4,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_utterances: int = 20,
        max_length: int = 128,
    ):
        """
        Initialize Hierarchical Transformer model.

        Args:
            model_name: HuggingFace model name for utterance encoder
            num_labels: Number of output classes
            d_model: Dimension of utterance embeddings
            n_heads: Number of attention heads in utterance transformer
            n_layers: Number of utterance transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_utterances: Max utterances in sequence
            max_length: Max tokens per utterance
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_utterances = max_utterances
        self.max_length = max_length

        # Utterance encoder (token-level BERT)
        self.utterance_encoder = UtteranceEncoder(model_name, dropout)
        self.bert_hidden = self.utterance_encoder.hidden_size

        # Positional encoding for utterance sequence
        self.pos_encoding = PositionalEncoding(d_model, max_utterances)

        # Utterance-level Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.utterance_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Multi-head attention for visualization/interpretability
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Classification head with pooling
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_labels)
        )

        self.dropout = nn.Dropout(dropout)

        # Learnable query for global attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.xavier_uniform_(self.query)

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
            utterance_mask: (batch, max_utterances) - mask for padding utterances

        Returns:
            Dictionary with logits, attention weights, and utterance embeddings
        """
        batch_size, max_utterances, max_length = input_ids.shape

        # Reshape for BERT processing
        input_ids_flat = input_ids.view(batch_size * max_utterances, max_length)
        attention_mask_flat = attention_mask.view(batch_size * max_utterances, max_length)

        # Encode each utterance with BERT
        utterance_embeddings = self.utterance_encoder(
            input_ids_flat,
            attention_mask_flat
        )  # (batch * max_utterances, d_model)

        # Reshape back to sequence format
        utterance_embeddings = utterance_embeddings.view(
            batch_size, max_utterances, -1
        )  # (batch, max_utterances, d_model)

        # Add positional encoding
        utterance_embeddings = self.pos_encoding(utterance_embeddings)
        utterance_embeddings = self.dropout(utterance_embeddings)

        # Create source key padding mask for Transformer
        # Transformer expects True for positions to ignore
        src_key_padding_mask = None
        if utterance_mask is not None:
            # utterance_mask: 1 for real, 0 for padding
            # src_key_padding_mask: True for padding (to ignore)
            src_key_padding_mask = (utterance_mask == 0)

        # Apply utterance-level Transformer
        transformer_output = self.utterance_transformer(
            utterance_embeddings,
            src_key_padding_mask=src_key_padding_mask,
        )  # (batch, max_utterances, d_model)

        # Global attention pooling using learnable query
        query_expanded = self.query.expand(batch_size, -1, -1)
        pooled_output, attention_weights = self.cross_attention(
            query=query_expanded,
            key=transformer_output,
            value=transformer_output,
            key_padding_mask=src_key_padding_mask,
        )
        pooled_output = pooled_output.squeeze(1)  # (batch, d_model)

        # Classification
        pooled = self.pooler(pooled_output)
        logits = self.classifier(pooled)  # (batch, num_labels)

        return {
            "logits": logits,
            "attention_weights": attention_weights.squeeze(1),  # (batch, max_utterances)
            "utterance_embeddings": utterance_embeddings,  # (batch, max_utterances, d_model)
        }

    def predict(self, **kwargs) -> torch.Tensor:
        """Make predictions."""
        output = self.forward(**kwargs)
        return torch.argmax(output["logits"], dim=1)


class HierarchicalPredictor:
    """Predictor wrapper for HierarchicalTransformerClassifier."""

    LABEL_NAMES = ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]

    def __init__(
        self,
        model: HierarchicalTransformerClassifier,
        tokenizer: AutoTokenizer,
        label_names: Optional[list] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_names = label_names or self.LABEL_NAMES
        self.model.eval()

    @torch.no_grad()
    def predict_single(
        self,
        utterances: list[str],
        device: str = "cpu",
    ) -> Dict[str, any]:
        """
        Predict label for a single sequence of utterances.

        Args:
            utterances: List of utterance strings
            device: Device to run on

        Returns:
            Dictionary with prediction, probabilities, and attention weights
        """
        self.model.to(device)

        # Tokenize
        encoded = self.tokenizer(
            utterances,
            padding="max_length",
            truncation=True,
            max_length=self.model.max_length,
            return_tensors="pt",
        )

        # Add batch dimension
        input_ids = encoded["input_ids"].unsqueeze(0).to(device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
        utterance_mask = torch.ones(1, len(utterances)).to(device)

        # Forward
        output = self.model(input_ids, attention_mask, utterance_mask)

        # Get prediction
        probs = torch.softmax(output["logits"], dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

        return {
            "prediction": self.label_names[pred_idx],
            "confidence": confidence,
            "probabilities": {
                self.label_names[i]: probs[i].item()
                for i in range(len(self.label_names))
            },
            "attention_weights": output["attention_weights"][0][:len(utterances)].cpu().numpy(),
        }


def create_model(
    model_name: str = "bert-base-uncased",
    num_labels: int = 4,
    **kwargs
) -> HierarchicalTransformerClassifier:
    """Factory function to create model."""
    return HierarchicalTransformerClassifier(
        model_name=model_name,
        num_labels=num_labels,
        **kwargs
    )


def create_tokenizer(model_name: str = "bert-base-uncased") -> AutoTokenizer:
    """Factory function to create tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    # Test the model
    print("Creating Hierarchical Transformer model...")
    model = create_model()
    tokenizer = create_tokenizer()

    # Test forward pass
    batch_size = 2
    max_utterances = 10
    max_length = 128

    input_ids = torch.randint(0, 30522, (batch_size, max_utterances, max_length))
    attention_mask = torch.ones(batch_size, max_utterances, max_length)

    print("Running forward pass...")
    output = model(input_ids, attention_mask)
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Attention weights shape: {output['attention_weights'].shape}")
    print(f"Utterance embeddings shape: {output['utterance_embeddings'].shape}")

    # Test predictor
    print("\nTesting predictor...")
    predictor = HierarchicalPredictor(model, tokenizer)

    utterances = [
        "Cleared for takeoff runway two seven left.",
        "Setting takeoff thrust.",
        "V1, rotate.",
        "Positive rate, gear up.",
        "After takeoff checklist complete.",
    ]

    result = predictor.predict_single(utterances)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
