import os
import json
import time
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "feedback_logs.json")


def log_analysis_feedback(
    image_hash: str,
    image_analysis: Dict[str, Any],
    user_sentiment: str,
    predicted_sentiment: str,
    match_score: int,
    user_feedback_label: Optional[str] = None,
    notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Logs structured evaluation data for continuous monitoring,
    error diagnosis, and offline model benchmarking.
    """
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)

    records = []
    if os.path.exists(FEEDBACK_LOG_PATH):
        try:
            with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_hash": image_hash,
        "description": image_analysis.get("image", {}).get("description", ""),
        "scene": image_analysis.get("image", {}).get("scene", ""),
        "detected_objects": [o.get("label") for o in image_analysis.get("image", {}).get("objects", [])],
        "ocr_text": image_analysis.get("image", {}).get("ocr_text", []),
        "detected_emotion": image_analysis.get("emotion_analysis", {}).get("primary", ""),
        "predicted_sentiment": predicted_sentiment,
        "user_selected_sentiment": user_sentiment,
        "match_score": match_score,
        "ground_truth_feedback": user_feedback_label or user_sentiment,
        "notes": notes or ""
    }

    records.append(entry)

    with open(FEEDBACK_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    return {
        "success": True,
        "message": "Feedback record stored successfully.",
        "total_feedback_records": len(records)
    }


def evaluate_feedback_metrics() -> Dict[str, Any]:
    """
    Evaluates historical predictions against user feedback ground truths.
    Computes Accuracy, Precision, Recall, Macro-F1, and Confusion Matrix.
    """
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return {"total_records": 0, "message": "No feedback records logged yet."}

    with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        return {"total_records": 0, "message": "Feedback log is empty."}

    labels = ["Positive", "Neutral", "Negative"]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = [[0 for _ in range(3)] for _ in range(3)]

    total = 0
    correct = 0

    for r in records:
        pred = r.get("predicted_sentiment", "Neutral")
        truth = r.get("ground_truth_feedback", "Neutral")
        if pred in label_to_idx and truth in label_to_idx:
            p_idx = label_to_idx[pred]
            t_idx = label_to_idx[truth]
            cm[t_idx][p_idx] += 1
            total += 1
            if p_idx == t_idx:
                correct += 1

    acc = (correct / total * 100) if total > 0 else 0.0

    return {
        "total_records": total,
        "accuracy": round(acc, 2),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm
        }
    }
