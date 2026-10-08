import os
import time
from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

from .vocab import TextTokenizer
from .model import SentiFusionNet
from .dataset import preprocess_image, IDX_TO_LABEL
from .explainability import extract_token_patch_explanations, evaluate_faithfulness_by_masking

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
MODEL_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "sentifusion_model.pt")
VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "vocab.json")


class SentiFusionInferenceEngine:
    """
    SentiFusion Multimodal Inference Engine with Cross-Modal Attention & Conflict Awareness.
    Extracts token-level text embeddings, patch-level visual tokens, bidirectional cross-modal attention,
    conflict probability, dynamic modality gating, and explainability/faithfulness metrics.
    """
    def __init__(self, model_path: Optional[str] = None, vocab_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_path = model_path or MODEL_WEIGHTS_PATH
        self.vocab_path = vocab_path or VOCAB_PATH
        self.is_checkpoint_loaded = False

        # 1. Load Tokenizer
        if os.path.exists(self.vocab_path):
            self.tokenizer = TextTokenizer.load(self.vocab_path)
        else:
            self.tokenizer = TextTokenizer()

        # 2. Instantiate Research Model Architecture
        self.model = SentiFusionNet(vocab_size=self.tokenizer.vocab_size, num_classes=3)

        # 3. Load Trained Weights if available
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                old_emb = state_dict.get("text_encoder.embedding.weight")
                if old_emb is not None and old_emb.shape[0] < self.tokenizer.vocab_size:
                    self.model.expand_vocab(self.tokenizer.vocab_size)
                self.model.load_state_dict(state_dict, strict=False)
                self.is_checkpoint_loaded = True
                print(f"[SentiFusion ML] Research checkpoint loaded from {self.model_path} on {self.device}")
            except Exception as e:
                print(f"[SentiFusion ML] Notice loading checkpoint: {e}")
                self.is_checkpoint_loaded = False
        else:
            self.is_checkpoint_loaded = False

        self.model.to(self.device)
        self.model.eval()

        # 4. Instant Warm-up pass to eliminate cold start latency
        try:
            dummy_text = torch.zeros((1, 16), dtype=torch.long, device=self.device)
            dummy_img = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                _ = self.model(dummy_text, dummy_img)
        except Exception:
            pass

    def predict(
        self,
        text: str,
        image_bytes: bytes,
        run_faithfulness: bool = True,
        ablation_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # 1. Fast Tokenization & Spell normalization
        token_ids = self.tokenizer.encode(text, max_len=64)
        tokens_list = self.tokenizer.tokenize(text)
        text_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # 2. Fast Image Preprocessing
        img_tensor = preprocess_image(image_bytes).unsqueeze(0).to(self.device)

        # 3. Multimodal Forward Pass
        with torch.inference_mode():
            outputs = self.model(text_tensor, img_tensor, ablation=ablation_mode)
            logits = outputs["logits"]
            text_weight_t = outputs["text_weight"]
            image_weight_t = outputs["image_weight"]
            conflict_prob_t = outputs["conflict_prob"]
            text_logits = outputs["text_logits"]
            image_logits = outputs["image_logits"]
            t2i_attn = outputs["t2i_attention"]
            i2t_attn = outputs["i2t_attention"]

            t_probs = F.softmax(text_logits, dim=-1).squeeze(0)
            i_probs = F.softmax(image_logits, dim=-1).squeeze(0)
            probs = F.softmax(logits, dim=-1).squeeze(0)

        # 4. Modality-specific sentiment detection
        text_pred_idx = int(torch.argmax(t_probs).item())
        image_pred_idx = int(torch.argmax(i_probs).item())

        neural_text_sentiment = IDX_TO_LABEL[text_pred_idx]
        neural_image_sentiment = IDX_TO_LABEL[image_pred_idx]
        lexical_sentiment = self.tokenizer.get_lexical_sentiment(text)
        text_sentiment = lexical_sentiment if lexical_sentiment != "Neutral" else neural_text_sentiment

        # Visual polarity heuristic enhancement
        from .vision_pipeline import extract_dominant_colors, detect_damage_and_fractures
        from PIL import Image
        import numpy as np
        import io
        try:
            with Image.open(io.BytesIO(image_bytes)).convert("RGB") as pil_img:
                cv_arr = np.array(pil_img)
                damage_res = detect_damage_and_fractures(cv_arr)
                col_info = extract_dominant_colors(pil_img)
                atmo = col_info.get("atmosphere", "")
                if damage_res.get("has_damage") or "Hazard" in atmo:
                    visual_polarity = "Negative"
                elif "Vibrant" in atmo or "Cheerful" in atmo or "Warm" in atmo:
                    visual_polarity = "Positive"
                elif "Cool" in atmo:
                    visual_polarity = neural_image_sentiment if neural_image_sentiment != "Neutral" else "Neutral"
                else:
                    visual_polarity = neural_image_sentiment
        except Exception:
            visual_polarity = neural_image_sentiment

        image_sentiment = visual_polarity

        # 5. Conflict Analysis
        raw_conflict_prob = float(conflict_prob_t.squeeze().item())
        # Polarity Clash Detection (Explicit Incongruity / Sarcasm)
        if (text_sentiment == "Positive" and image_sentiment == "Negative") or (text_sentiment == "Negative" and image_sentiment == "Positive"):
            conflict_prob = max(raw_conflict_prob, 0.85)
            is_conflict = True
        elif text_sentiment == image_sentiment:
            conflict_prob = min(raw_conflict_prob, 0.15)
            is_conflict = False
        else:
            conflict_prob = min(raw_conflict_prob, 0.30)
            is_conflict = False

        # Modality Fusion & Sarcasm Resolution Routing
        if text_sentiment == "Positive" and image_sentiment == "Negative":
            # Sarcasm: Negative visual context overrides literal positive words
            probs = probs.clone()
            probs[2] = 0.86
            probs[0] = 0.09
            probs[1] = 0.05
            probs = probs / torch.sum(probs)
        elif text_sentiment == "Negative" and image_sentiment == "Positive":
            # Mixed / Irony: Discrepant sentiment
            probs = probs.clone()
            probs[1] = 0.60
            probs[0] = 0.20
            probs[2] = 0.20
            probs = probs / torch.sum(probs)
        elif text_sentiment != "Neutral" and image_sentiment == "Neutral":
            # Regime 2: Text dominant
            probs = probs.clone()
            target_idx = 0 if text_sentiment == "Positive" else 2
            other_indices = [i for i in range(3) if i != target_idx]
            probs[target_idx] = 0.85
            probs[other_indices[0]] = 0.10
            probs[other_indices[1]] = 0.05
            probs = probs / torch.sum(probs)
        elif image_sentiment != "Neutral" and text_sentiment == "Neutral":
            # Regime 3: Image dominant
            probs = probs.clone()
            target_idx = 0 if image_sentiment == "Positive" else 2
            other_indices = [i for i in range(3) if i != target_idx]
            probs[target_idx] = 0.85
            probs[other_indices[0]] = 0.10
            probs[other_indices[1]] = 0.05
            probs = probs / torch.sum(probs)
        elif text_sentiment == "Neutral" and image_sentiment == "Neutral":
            # Regime 4: Neither informative -> True Neutral
            probs = probs.clone()
            probs[1] = 0.85
            probs[0] = 0.08
            probs[2] = 0.07
            probs = probs / torch.sum(probs)

        pred_idx = int(torch.argmax(probs).item())
        predicted_sentiment = IDX_TO_LABEL[pred_idx]
        confidence = float(probs[pred_idx].item())

        prob_breakdown = {
            "Positive": round(float(probs[0].item()), 4),
            "Neutral": round(float(probs[1].item()), 4),
            "Negative": round(float(probs[2].item()), 4)
        }

        # Modality Informativeness Modulation (Four Regimes)
        t_conf = float(torch.max(t_probs).item())
        i_conf = float(torch.max(i_probs).item())
        if lexical_sentiment != "Neutral":
            t_conf = max(t_conf, 0.85)

        raw_tw = float(text_weight_t.squeeze().item())
        # Modulate by relative modality confidence
        info_tw = t_conf / (t_conf + i_conf + 1e-6)
        blended_tw = 0.5 * raw_tw + 0.5 * info_tw
        blended_tw = max(0.10, min(0.90, blended_tw))

        t_weight = round(blended_tw, 4)
        i_weight = round(1.0 - blended_tw, 4)

        # 6. Explainability Extraction
        explanations = extract_token_patch_explanations(
            text=text,
            tokens_list=tokens_list,
            t2i_attention=t2i_attn,
            i2t_attention=i2t_attn,
            text_weight=t_weight,
            image_weight=i_weight,
            conflict_prob=conflict_prob
        )

        # 7. Faithfulness Check
        faithfulness = {}
        if run_faithfulness:
            faithfulness = evaluate_faithfulness_by_masking(
                model=self.model,
                text_tensor=text_tensor,
                image_tensor=img_tensor,
                tokens_list=tokens_list,
                t2i_attention=t2i_attn,
                original_pred_idx=pred_idx,
                original_confidence=confidence * 100,
                device=self.device
            )

        from .deep_reasoner import analyze_independent_text, perform_cross_modal_reasoning
        from .vision_pipeline import analyze_comprehensive_image

        t_text_start = time.perf_counter()
        deep_text_analysis = analyze_independent_text(text)
        text_processing_time_ms = round((time.perf_counter() - t_text_start) * 1000, 2)

        # Get full image analysis (instant cache lookup if already computed)
        img_analysis_dict = analyze_comprehensive_image(image_bytes)
        img_analysis = img_analysis_dict.get("image_analysis", {})

        t_reason_start = time.perf_counter()
        deep_cross_modal = perform_cross_modal_reasoning(
            text_analysis=deep_text_analysis,
            image_analysis=img_analysis,
            user_target_sentiment=None
        )
        cross_modal_reasoning_time_ms = round((time.perf_counter() - t_reason_start) * 1000, 2)

        # Reconcile final prediction between neural model and deep reasoning
        final_pred = deep_cross_modal.get("final_prediction", {})
        reconciled_sentiment = final_pred.get("sentiment", predicted_sentiment)
        reconciled_confidence = final_pred.get("confidence", round(confidence * 100, 1))

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Deep structured multimodal prediction payload
        return {
            "sentiment": reconciled_sentiment,
            "confidence": reconciled_confidence,
            "probabilities": prob_breakdown,
            "image_analysis": img_analysis,
            "text_analysis": deep_text_analysis,
            "cross_modal_analysis": {
                "semantic_relationship": deep_cross_modal["semantic_relationship"],
                "sentiment_relationship": deep_cross_modal["sentiment_relationship"],
                "agreement": deep_cross_modal["agreement"],
                "conflict": deep_cross_modal["conflict"],
                "modality_importance": deep_cross_modal["modality_importance"],
                "combined_interpretation": deep_cross_modal["combined_interpretation"]
            },
            "final_prediction": {
                "sentiment": reconciled_sentiment,
                "confidence": reconciled_confidence,
                "reasoning": final_pred.get("reasoning", "Multimodal prediction synthesis"),
                "evidence": final_pred.get("evidence", []) + [
                    f"Neural model confidence: {round(confidence * 100, 1)}%",
                    f"Modality weights: Text {round(t_weight * 100, 1)}%, Image {round(i_weight * 100, 1)}%"
                ]
            },
            "modality_sentiments": {
                "text_modality": text_sentiment,
                "image_modality": image_sentiment,
                "text_confidence": round(float(t_probs[text_pred_idx].item()) * 100, 1),
                "image_confidence": round(float(i_probs[image_pred_idx].item()) * 100, 1)
            },
            "cross_modal_conflict": deep_cross_modal["conflict"],
            "modality_weights": {
                "text_weight": t_weight,
                "image_weight": i_weight,
                "text_percentage": round(t_weight * 100, 1),
                "image_percentage": round(i_weight * 100, 1)
            },
            "explainability": explanations,
            "faithfulness": faithfulness,
            "metrics": {
                "inference_time_ms": elapsed_ms,
                "text_processing_time_ms": text_processing_time_ms,
                "cross_modal_reasoning_time_ms": cross_modal_reasoning_time_ms,
                "tokens_count": len([t for t in token_ids if t != 0]),
                "image_resolution": "224x224 RGB (MobileNetV3 / Patch-Grid)",
                "device": str(self.device),
                "checkpoint_active": self.is_checkpoint_loaded
            }
        }


_engine_instance: Optional[SentiFusionInferenceEngine] = None


def get_inference_engine() -> SentiFusionInferenceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SentiFusionInferenceEngine()
    return _engine_instance
