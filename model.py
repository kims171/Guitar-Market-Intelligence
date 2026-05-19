"""
model.py — Multi-Modal PyTorch Price Prediction Model

Architecture:
  - Text branch:   DistilBERT → 768-dim embedding → projection head
  - Tabular branch: MLP over numerical + one-hot features
  - Fusion:         Concatenate both branches → shared MLP → price prediction

The multi-modal design mirrors the proposal: integrating structured tabular
sales data with unstructured text (listing descriptions) for richer signals.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    DistilBERT encoder for listing descriptions.
    Uses the [CLS] token embedding as the sentence representation.
    """

    def __init__(self, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Freeze first 4 transformer layers — fine-tune only the top 2
        for i, layer in enumerate(self.bert.transformer.layer):
            if i < 4:
                for param in layer.parameters():
                    param.requires_grad = False

        bert_hidden = 768  # DistilBERT hidden size
        self.projection = nn.Sequential(
            nn.Linear(bert_hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            text_embedding: (batch, output_dim)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] representation
        return self.projection(cls_token)


class TabularEncoder(nn.Module):
    """
    MLP over structured tabular features:
    year_of_manufacture, originality_score, is_player_grade,
    one-hot brand, one-hot condition, one-hot pickup config.
    """

    def __init__(self, input_dim: int, output_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) — normalized tabular features
        Returns:
            tabular_embedding: (batch, output_dim)
        """
        return self.mlp(x)


class FusionHead(nn.Module):
    """
    Concatenates text and tabular embeddings, then regresses to a price prediction.
    Also outputs uncertainty (log-variance) for confidence intervals.
    """

    def __init__(self, text_dim: int = 128, tabular_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        fused_dim = text_dim + tabular_dim

        self.shared = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # Mean prediction head
        self.mean_head = nn.Linear(128, 1)

        # Log-variance head → uncertainty estimate (used for confidence intervals)
        self.log_var_head = nn.Linear(128, 1)

    def forward(
        self, text_emb: torch.Tensor, tabular_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean:    (batch, 1) — predicted price
            log_var: (batch, 1) — predicted log-variance (uncertainty)
        """
        fused = torch.cat([text_emb, tabular_emb], dim=-1)
        shared = self.shared(fused)
        mean = self.mean_head(shared)
        log_var = self.log_var_head(shared)
        return mean, log_var


# ---------------------------------------------------------------------------
# Full Multi-Modal Model
# ---------------------------------------------------------------------------

class GuitarPriceModel(nn.Module):
    """
    Multi-modal model integrating:
      - DistilBERT text encoder (listing description)
      - Tabular MLP encoder (structured guitar features)
    """

    def __init__(self, tabular_input_dim: int, text_output_dim: int = 128, tabular_output_dim: int = 64):
        super().__init__()
        self.text_encoder = TextEncoder(output_dim=text_output_dim)
        self.tabular_encoder = TabularEncoder(input_dim=tabular_input_dim, output_dim=tabular_output_dim)
        self.fusion = FusionHead(text_dim=text_output_dim, tabular_dim=tabular_output_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:        (batch, seq_len)
            attention_mask:   (batch, seq_len)
            tabular_features: (batch, tabular_input_dim)
        Returns:
            mean_price: (batch, 1)
            log_var:    (batch, 1)
        """
        text_emb = self.text_encoder(input_ids, attention_mask)
        tab_emb = self.tabular_encoder(tabular_features)
        return self.fusion(text_emb, tab_emb)


# ---------------------------------------------------------------------------
# Loss: Gaussian Negative Log-Likelihood
# Trains the model to predict both the price AND its uncertainty simultaneously.
# ---------------------------------------------------------------------------

class GaussianNLLLoss(nn.Module):
    """
    Negative log-likelihood loss for a Gaussian distribution.
    Encourages the model to be calibrated — not just accurate.
    Loss = 0.5 * (log_var + (pred - target)^2 / exp(log_var))
    """

    def forward(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        precision = torch.exp(-log_var)
        return torch.mean(0.5 * (log_var + precision * (mean - target) ** 2))


# ---------------------------------------------------------------------------
# Tokenizer factory (shared with dataset & inference)
# ---------------------------------------------------------------------------

def get_tokenizer() -> DistilBertTokenizerFast:
    return DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
