"""
train.py — Training & Evaluation Pipeline

Usage:
    python train.py

Saves best model checkpoint to checkpoints/best_model.pt
Logs train/val loss and MAE to console.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast

from model import GaussianNLLLoss, GuitarPriceModel, get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_PATH = Path("data/processed_listings.csv")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 2e-5
MAX_TEXT_LEN = 256      # Max token length for listing description
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1         # Remaining 0.1 → test

# Tabular feature columns (after one-hot encoding in preprocess.py)
BASE_TABULAR_COLS = [
    "year_of_manufacture",
    "originality_score",
    "is_player_grade",
]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GuitarListingDataset(Dataset):
    """
    Wraps processed guitar listings.
    Returns tokenized description + tabular features + log-price target.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tabular_cols: list[str],
        tokenizer: DistilBertTokenizerFast,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.tabular_cols = tabular_cols

        # Log-transform the target (price) → more stable training
        self.targets = np.log1p(df["price_usd_normalized"].values).astype(np.float32)

        # Scale tabular features
        tab = df[tabular_cols].fillna(0).values.astype(np.float32)
        if fit_scaler:
            self.scaler = StandardScaler()
            self.tabular = self.scaler.fit_transform(tab)
        else:
            assert scaler is not None, "Must provide a fitted scaler for val/test."
            self.scaler = scaler
            self.tabular = scaler.transform(tab)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        description = str(self.df.at[idx, "description"])

        encoding = self.tokenizer(
            description,
            max_length=MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "tabular": torch.tensor(self.tabular[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Training & Evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        mean, log_var = model(input_ids, attention_mask, tabular)
        loss = loss_fn(mean, log_var, targets)
        loss.backward()

        # Gradient clipping — important with BERT fine-tuning
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    """Returns (avg_loss, mae_in_dollars)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)

        mean, log_var = model(input_ids, attention_mask, tabular)
        loss = loss_fn(mean, log_var, targets)
        total_loss += loss.item()

        # Convert log-price back to dollars for interpretable MAE
        preds_usd = torch.expm1(mean).cpu().numpy()
        targets_usd = torch.expm1(targets).cpu().numpy()
        all_preds.extend(preds_usd.flatten())
        all_targets.extend(targets_usd.flatten())

    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    return total_loss / len(loader), mae


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training():
    logger.info(f"Using device: {DEVICE}")

    # --- Load data ---
    df = pd.read_csv(PROCESSED_PATH)
    logger.info(f"Loaded {len(df)} rows.")

    # Identify one-hot encoded columns dynamically
    ohe_cols = [c for c in df.columns if c.startswith(("brand_", "condition_normalized_", "pickup_config_"))]
    tabular_cols = BASE_TABULAR_COLS + ohe_cols
    logger.info(f"Tabular feature count: {len(tabular_cols)}")

    # --- Train / Val / Test split ---
    train_df, temp_df = train_test_split(df, test_size=1 - TRAIN_SPLIT, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    logger.info(f"Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    tokenizer = get_tokenizer()

    train_ds = GuitarListingDataset(train_df, tabular_cols, tokenizer, fit_scaler=True)
    val_ds = GuitarListingDataset(val_df, tabular_cols, tokenizer, scaler=train_ds.scaler)
    test_ds = GuitarListingDataset(test_df, tabular_cols, tokenizer, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model ---
    model = GuitarPriceModel(tabular_input_dim=len(tabular_cols)).to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Differential learning rates: lower LR for BERT, higher for task heads
    optimizer = torch.optim.AdamW([
        {"params": model.text_encoder.bert.parameters(), "lr": 1e-5},
        {"params": model.text_encoder.projection.parameters(), "lr": LEARNING_RATE},
        {"params": model.tabular_encoder.parameters(), "lr": LEARNING_RATE},
        {"params": model.fusion.parameters(), "lr": LEARNING_RATE},
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = GaussianNLLLoss()

    best_val_loss = math.inf
    best_checkpoint = CHECKPOINT_DIR / "best_model.pt"

    # --- Training loop ---
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_mae = evaluate(model, val_loader, loss_fn, DEVICE)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: ${val_mae:,.0f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "tabular_cols": tabular_cols,
                    "scaler": train_ds.scaler,
                },
                best_checkpoint,
            )
            logger.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")

    # --- Test evaluation ---
    checkpoint = torch.load(best_checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_mae = evaluate(model, test_loader, loss_fn, DEVICE)
    logger.info(f"\nTest MAE: ${test_mae:,.0f}  |  Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    run_training()
