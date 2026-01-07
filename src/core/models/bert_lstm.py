"""
Model A: BERT + LSTM for sequential CVR analysis.

Architecture:
- Per-utterance BERT encoding
- Bi-LSTM for sequential dependencies
- Attention mechanism
- Classification head
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BertLSTMClassifier(nn.Module):
    """
    BERT + LSTM model for sequential anomaly detection.

    Processes a sequence of utterances and predicts temporal labels.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 4,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        max_utterances: int = 20,
        max_length: int = 128,
    ):
        """
        Initialize BERT+LSTM model.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            max_utterances: Max utterances in sequence
            max_length: Max tokens per utterance
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.lstm_hidden = lstm_hidden
        self.max_utterances = max_utterances
        self.max_length = max_length

        # BERT encoder for utterances
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_hidden = self.bert.config.hidden_size

        # Bi-LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=self.bert_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )

        self.dropout = nn.Dropout(dropout)

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
            Dictionary with logits and attention weights
        """
        batch_size, max_utterances, max_length = input_ids.shape

        # Reshape for BERT processing
        input_ids_flat = input_ids.view(batch_size * max_utterances, max_length)
        attention_mask_flat = attention_mask.view(batch_size * max_utterances, max_length)

        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
        )
        # Use [CLS] token representation
        utterance_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch * n_utter, hidden)
        utterance_embeddings = utterance_embeddings.view(
            batch_size, max_utterances, -1
        )  # (batch, n_utter, hidden)

        utterance_embeddings = self.dropout(utterance_embeddings)

        # LSTM processing
        lstm_out, _ = self.lstm(utterance_embeddings)  # (batch, n_utter, hidden*2)

        # Attention over utterances
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch, n_utter)

        # Apply utterance mask if provided
        if utterance_mask is not None:
            attention_scores = attention_scores.masked_fill(
                utterance_mask == 0, float("-inf")
            )

        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, n_utter)

        # Weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (batch, hidden*2)

        # Classification
        logits = self.classifier(context)  # (batch, num_labels)

        return {
            "logits": logits,
            "attention_weights": attention_weights,
        }

    def predict(self, **kwargs) -> torch.Tensor:
        """Make predictions."""
        output = self.forward(**kwargs)
        return torch.argmax(output["logits"], dim=1)


class BertLSTMPredictor:
    """Predictor wrapper for BertLSTMClassifier."""

    LABEL_NAMES = ["NORMAL", "EARLY_WARNING", "ELEVATED", "CRITICAL"]

    def __init__(
        self,
        model: BertLSTMClassifier,
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
            Dictionary with prediction and metadata
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
) -> BertLSTMClassifier:
    """Factory function to create model."""
    return BertLSTMClassifier(
        model_name=model_name,
        num_labels=num_labels,
        **kwargs
    )


def create_tokenizer(model_name: str = "bert-base-uncased") -> AutoTokenizer:
    """Factory function to create tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    # Test the model
    print("Creating model...")
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

    # Test predictor
    print("\nTesting predictor...")
    predictor = BertLSTMPredictor(model, tokenizer)

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
