import os
import sys
import json
import random
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.vocab import TextTokenizer
from ml.model import SentiFusionNet
from ml.dataset import MultimodalDataset, IDX_TO_LABEL, LABEL_TO_IDX


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Stratified split into Train, Validation, and Test partitions.
    """
    random.seed(seed)
    by_class: Dict[str, List[Dict]] = {}
    for s in samples:
        by_class.setdefault(s["label"], []).append(s)

    train, val, test = [], [], []
    for cls, items in by_class.items():
        shuffled = items.copy()
        random.shuffle(shuffled)
        n_train = int(len(shuffled) * train_ratio)
        n_val = int(len(shuffled) * val_ratio)

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    return train, val, test


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int = 3) -> Dict:
    """
    Computes Accuracy, Precision, Recall, Macro F1, and Confusion Matrix.
    """
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    total = len(y_true)
    correct = sum(cm[i][i] for i in range(num_classes))
    accuracy = correct / total if total > 0 else 0.0

    precisions, recalls, f1s = [], [], []
    for i in range(num_classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes)) - tp
        fn = sum(cm[i][c] for c in range(num_classes)) - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_f1 = sum(f1s) / num_classes
    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes

    return {
        "accuracy": round(accuracy * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "macro_precision": round(macro_precision * 100, 2),
        "macro_recall": round(macro_recall * 100, 2),
        "per_class": {
            IDX_TO_LABEL[i]: {
                "precision": round(precisions[i] * 100, 2),
                "recall": round(recalls[i] * 100, 2),
                "f1": round(f1s[i] * 100, 2)
            }
            for i in range(num_classes)
        },
        "confusion_matrix": cm
    }


def evaluate_model_mode(
    model: SentiFusionNet,
    loader: DataLoader,
    device: torch.device,
    mode: str = "multimodal"
) -> Dict:
    """
    Evaluate in multimodal, text-only (zeroed vision), or image-only (zeroed text) modes.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for texts, imgs, labels in loader:
            texts = texts.to(device)
            imgs = imgs.to(device)

            if mode == "text_only":
                imgs = torch.zeros_like(imgs)
            elif mode == "image_only":
                texts = torch.zeros_like(texts)

            outputs = model(texts, imgs)
            logits = outputs[0]
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_true.extend(labels.tolist())
            y_pred.extend(preds)

    return compute_metrics(y_true, y_pred)


def print_confusion_matrix(cm: List[List[int]]):
    classes = [IDX_TO_LABEL[i] for i in range(len(cm))]
    header = f"{'Actual \\ Pred':<15}" + "".join(f"{c:>12}" for c in classes)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{classes[i]:<15}" + "".join(f"{val:>12}" for val in row)
        print(row_str)


def run_evaluation():
    print("=" * 60)
    print(" SentiFusion Comprehensive Evaluation & Ablation Suite")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.json")
    img_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found at {data_path}. Run `python data/create_sample_dataset.py` first.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # 1. Stratified Partitioning
    train_samples, val_samples, test_samples = split_dataset(samples, train_ratio=0.70, val_ratio=0.15)
    print(f"[Data Partition] Total: {len(samples)} | Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # 2. Tokenizer & Loaders
    vocab_file = os.path.join(ckpt_dir, "vocab.json")
    if os.path.exists(vocab_file):
        tokenizer = TextTokenizer.load(vocab_file)
        tokenizer.build_vocab_from_texts([s["text"] for s in samples])
    else:
        tokenizer = TextTokenizer()
        tokenizer.build_vocab_from_texts([s["text"] for s in samples])
    
    test_dataset = MultimodalDataset(test_samples, tokenizer=tokenizer, image_root_dir=img_root)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Model setup
    model = SentiFusionNet(vocab_size=tokenizer.vocab_size, num_classes=3)
    model_path = os.path.join(ckpt_dir, "sentifusion_model.pt")
    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
            old_emb = state.get("text_encoder.embedding.weight")
            if old_emb is not None and old_emb.shape[0] < tokenizer.vocab_size:
                model.expand_vocab(tokenizer.vocab_size)
            model.load_state_dict(state, strict=False)
            print(f"[Checkpoints] Successfully loaded trained weights from {model_path}")
        except Exception as e:
            print(f"[Checkpoints] Notice loading weights: {e}, using baseline model.")
    model.to(device)

    # 4. Ablation Studies
    modes = [
        ("Multimodal Fusion (Text + Image)", "multimodal"),
        ("Text-Only Modality Ablation", "text_only"),
        ("Image-Only Modality Ablation", "image_only")
    ]

    print("\n" + "=" * 60)
    print(" ABLATION STUDY RESULTS (Test Set)")
    print("=" * 60)

    for title, mode in modes:
        metrics = evaluate_model_mode(model, test_loader, device, mode=mode)
        print(f"\n--- {title} ---")
        print(f"Accuracy:        {metrics['accuracy']}%")
        print(f"Macro F1-Score:  {metrics['macro_f1']}%")
        print(f"Macro Precision: {metrics['macro_precision']}%")
        print(f"Macro Recall:    {metrics['macro_recall']}%")
        print("\nPer-Class Breakdown:")
        for cls_name, vals in metrics["per_class"].items():
            print(f"  {cls_name:<10} -> Precision: {vals['precision']:>5}%, Recall: {vals['recall']:>5}%, F1: {vals['f1']:>5}%")
        print("\nConfusion Matrix:")
        print_confusion_matrix(metrics["confusion_matrix"])

    print("\n" + "=" * 60)
    print(" Evaluation Completed Successfully.")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
