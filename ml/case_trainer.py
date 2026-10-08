import os
import sys
import json
import uuid
import io
from typing import Dict, Any, Optional
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.vocab import TextTokenizer
from ml.model import SentiFusionNet
from ml.dataset import MultimodalDataset, LABEL_MAP, IDX_TO_LABEL
import ml.inference as inference_module

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.json")
CASES_IMG_DIR = os.path.join(DATA_DIR, "cases")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "ml", "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "sentifusion_model.pt")
VOCAB_PATH = os.path.join(CHECKPOINTS_DIR, "vocab.json")


def train_on_new_case(
    text: str,
    image_bytes: bytes,
    label: str,
    epochs: int = 15,
    lr: float = 0.002
) -> Dict[str, Any]:
    """
    Ingests a new multimodal case (text + image + ground truth label),
    persists the image, expands vocabulary dynamically, fine-tunes the deep model,
    saves the updated weights, and hot-reloads the live inference engine.
    """
    label_norm = label.strip().capitalize()
    if label_norm not in LABEL_MAP:
        raise ValueError(f"Invalid label '{label}'. Must be one of: Positive, Neutral, Negative")

    os.makedirs(CASES_IMG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    # 1. Save Case Image securely
    case_id = uuid.uuid4().hex[:8]
    image_filename = f"case_{case_id}.jpg"
    image_disk_path = os.path.join(CASES_IMG_DIR, image_filename)
    relative_img_path = f"cases/{image_filename}"

    # Verify and normalize image with PIL before saving
    image_stream = io.BytesIO(image_bytes)
    with Image.open(image_stream) as pil_img:
        pil_img.convert("RGB").save(image_disk_path, format="JPEG", quality=92)

    # 2. Append new case to dataset.json
    samples = []
    if os.path.exists(DATASET_PATH):
        try:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                samples = json.load(f)
        except Exception:
            samples = []

    new_entry = {
        "text": text.strip(),
        "image": relative_img_path,
        "label": label_norm
    }
    samples.append(new_entry)

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    # 3. Update Tokenizer Vocabulary dynamically with new words from case
    tokenizer = TextTokenizer()
    if os.path.exists(VOCAB_PATH):
        try:
            tokenizer = TextTokenizer.load(VOCAB_PATH)
        except Exception:
            pass

    new_words_added = tokenizer.build_vocab_from_texts([text.strip()])
    tokenizer.build_vocab_from_texts([s["text"] for s in samples])
    tokenizer.save(VOCAB_PATH)

    # 4. Prepare Dataset and Weighted Training Loader
    dataset = MultimodalDataset(samples, tokenizer=tokenizer, image_root_dir=DATA_DIR)
    # Batch size adapts to dataset size
    batch_size = min(8, max(1, min(4, len(samples))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentiFusionNet(vocab_size=tokenizer.vocab_size, num_classes=3)

    # 5. Load existing weights and dynamically expand vocabulary dimension
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            old_emb = state_dict.get("text_encoder.embedding.weight")
            if old_emb is not None and old_emb.shape[0] != tokenizer.vocab_size:
                # Retain previous embeddings and dynamically assign to expanded matrix
                with torch.no_grad():
                    min_v = min(old_emb.shape[0], tokenizer.vocab_size)
                    model.text_encoder.embedding.weight[:min_v] = old_emb[:min_v]
                del state_dict["text_encoder.embedding.weight"]
            model.load_state_dict(state_dict, strict=False)
            print(f"[Continuous Learning] Loaded existing weights, updated vocab size to {tokenizer.vocab_size}.")
        except Exception as e:
            print(f"[Continuous Learning] Notice loading weights: {e}")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_loss = 0.0
    final_acc = 0.0

    # Repeat training epochs with mini-batches
    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for texts, imgs, targets in loader:
            texts, imgs, targets = texts.to(device), imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(texts, imgs)
            logits = out["logits"]
            text_logits = out["text_logits"]
            image_logits = out["image_logits"]
            loss_multi = criterion(logits, targets)
            loss_text = criterion(text_logits, targets)
            loss_img = criterion(image_logits, targets)
            loss = loss_multi + 0.3 * loss_text + 0.3 * loss_img

            loss.backward()
            optimizer.step()

            total_loss += loss_multi.item() * len(targets)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        final_acc = correct / total if total > 0 else 0.0
        final_loss = total_loss / total if total > 0 else 0.0

    # 6. Save updated checkpoint
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[Continuous Learning] Checkpoint updated at {MODEL_PATH} (Accuracy: {final_acc*100:.1f}%)")

    # 7. Hot-reload the in-memory inference engine
    inference_module._engine_instance = inference_module.SentiFusionInferenceEngine(
        model_path=MODEL_PATH,
        vocab_path=VOCAB_PATH
    )

    return {
        "success": True,
        "message": f"Successfully trained model on case with label '{label_norm}' ({new_words_added} new vocabulary words learned).",
        "dataset_total_samples": len(samples),
        "vocab_size": tokenizer.vocab_size,
        "new_words_learned": new_words_added,
        "training_accuracy": round(final_acc * 100, 2),
        "training_loss": round(final_loss, 4),
        "case_saved_path": relative_img_path
    }
