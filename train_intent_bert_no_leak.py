import argparse
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup


@dataclass
class SplitData:
    train_text: pd.Series
    val_text: pd.Series
    test_text: pd.Series
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_phrase(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def build_phrase_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    data = df[["phrase", "prompt"]].copy()
    data["phrase_norm"] = data["phrase"].map(normalize_phrase)

    # Keep one row per (phrase, label) pair to remove repeated recordings.
    unique_pairs = data.drop_duplicates(subset=["phrase_norm", "prompt"]).copy()

    # Remove ambiguous phrases that map to multiple labels.
    label_count_per_phrase = unique_pairs.groupby("phrase_norm")["prompt"].nunique()
    ambiguous_phrases = set(label_count_per_phrase[label_count_per_phrase > 1].index)

    clean = unique_pairs[~unique_pairs["phrase_norm"].isin(ambiguous_phrases)].copy()
    clean = clean.drop_duplicates(subset=["phrase_norm"]).reset_index(drop=True)

    print(f"Original rows: {len(df)}")
    print(f"Unique (phrase,label) pairs: {len(unique_pairs)}")
    print(f"Ambiguous phrases removed: {len(ambiguous_phrases)}")
    print(f"Final unique phrases used for training: {len(clean)}")

    return clean


def split_data(df_phrase_level: pd.DataFrame, seed: int) -> SplitData:
    X = df_phrase_level["phrase_norm"]
    y = df_phrase_level["prompt"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.125,  # 0.125 * 0.8 = 0.1 total
        random_state=seed,
        stratify=y_train_val,
    )

    train_set = set(X_train)
    val_set = set(X_val)
    test_set = set(X_test)
    print(f"Overlap train/val: {len(train_set & val_set)}")
    print(f"Overlap train/test: {len(train_set & test_set)}")
    print(f"Overlap val/test: {len(val_set & test_set)}")
    print(f"Train/Val/Test sizes: {len(X_train)} / {len(X_val)} / {len(X_test)}")

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def tokenize_text(tokenizer: BertTokenizer, texts: pd.Series) -> Dict[str, torch.Tensor]:
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


def build_loaders(
    tokenizer: BertTokenizer,
    split: SplitData,
    batch_size_train: int,
    batch_size_eval: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    le = LabelEncoder()
    y_train = torch.tensor(le.fit_transform(split.y_train), dtype=torch.long)
    y_val = torch.tensor(le.transform(split.y_val), dtype=torch.long)
    y_test = torch.tensor(le.transform(split.y_test), dtype=torch.long)

    tok_train = tokenize_text(tokenizer, split.train_text)
    tok_val = tokenize_text(tokenizer, split.val_text)
    tok_test = tokenize_text(tokenizer, split.test_text)

    train_dataset = TensorDataset(tok_train["input_ids"], tok_train["attention_mask"], y_train)
    val_dataset = TensorDataset(tok_val["input_ids"], tok_val["attention_mask"], y_val)
    test_dataset = TensorDataset(tok_test["input_ids"], tok_test["attention_mask"], y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False)

    return train_loader, val_loader, test_loader, le


def evaluate(model: BertForSequenceClassification, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc, np.array(all_labels), np.array(all_preds)


def train(
    model: BertForSequenceClassification,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    eps: float,
) -> None:
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            running_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = running_loss / max(1, len(train_loader))
        val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{epochs} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BERT intent classifier with leakage-resistant split.")
    parser.add_argument("--data", default="data.csv", help="Path to CSV with 'phrase' and 'prompt' columns")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-train", type=int, default=32)
    parser.add_argument("--batch-eval", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=1215)
    args = parser.parse_args()

    set_seed(args.seed)

    df = pd.read_csv(args.data)
    if not {"phrase", "prompt"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'phrase' and 'prompt'.")

    clean_df = build_phrase_level_dataset(df)
    split = split_data(clean_df, seed=args.seed)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    train_loader, val_loader, test_loader, le = build_loaders(
        tokenizer,
        split,
        batch_size_train=args.batch_train,
        batch_size_eval=args.batch_eval,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(le.classes_),
    )
    model.to(device)

    train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        eps=args.eps,
    )

    test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\nLeakage-safe test accuracy: {test_acc:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_, digits=4))


if __name__ == "__main__":
    main()
