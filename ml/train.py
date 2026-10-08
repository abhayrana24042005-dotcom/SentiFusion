import os
import sys
import json
from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.vocab import TextTokenizer
from ml.model import SentiFusionNet
from ml.dataset import MultimodalDataset

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_JSON_PATH = os.path.join(DATA_DIR, "dataset.json")


def train_sentifusion_model(
    dataset_samples: Optional[List[Dict]] = None,
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-3,
    feature_dim: int = 128,
    save_dir: str = CHECKPOINTS_DIR
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}")

    # Load dataset if not provided
    if dataset_samples is None:
        if os.path.exists(DATASET_JSON_PATH):
            with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
                dataset_samples = json.load(f)
        else:
            raise FileNotFoundError(f"Dataset not found at {DATASET_JSON_PATH}. Run `python data/create_sample_dataset.py` first.")

    print(f"[Training] Dataset contains {len(dataset_samples)} multimodal samples.")
    if len(dataset_samples) == 0:
        print("[Training] Dataset is currently empty. Waiting for user to add test cases.")
        return None

    # 1. Build and save vocabulary
    all_texts = [s["text"] for s in dataset_samples]
    tokenizer = TextTokenizer()
    tokenizer.build_vocab_from_texts(all_texts)
    vocab_path = os.path.join(save_dir, "vocab.json")
    tokenizer.save(vocab_path)
    print(f"[Training] Tokenizer vocabulary constructed ({tokenizer.vocab_size} tokens) and saved to {vocab_path}")

    # 2. Build Dataset & DataLoader
    dataset = MultimodalDataset(dataset_samples, tokenizer=tokenizer, image_root_dir=DATA_DIR)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Instantiate SentiFusionNet Architecture
    model = SentiFusionNet(vocab_size=tokenizer.vocab_size, num_classes=3, feature_dim=feature_dim)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_model_path = os.path.join(save_dir, "sentifusion_model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for texts, imgs, labels in train_loader:
            texts, imgs, labels = texts.to(device), imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(texts, imgs)
            logits = out["logits"]
            text_logits = out["text_logits"]
            image_logits = out["image_logits"]
            loss_multi = criterion(logits, labels)
            loss_text = criterion(text_logits, labels)
            loss_img = criterion(image_logits, labels)
            loss = loss_multi + 0.3 * loss_text + 0.3 * loss_img

            loss.backward()
            optimizer.step()

            total_loss += loss_multi.item() * texts.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Accuracy: {train_acc*100:.1f}%")

    torch.save(model.state_dict(), best_model_path)
    print(f"[Training Complete] Checkpoint successfully saved to: {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    train_sentifusion_model()
