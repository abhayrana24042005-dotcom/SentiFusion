import os
import sys
import io
import time
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.vocab import TextTokenizer
from ml.model import SentiFusionNet
from ml.dataset import preprocess_image
from ml.inference import SentiFusionInferenceEngine
from ml.vision_pipeline import analyze_comprehensive_image, _memory_cache
from ml.sentiment_matcher import calculate_sentiment_match


def create_mock_image(color=(34, 197, 94), pattern="circle") -> bytes:
    """Helper to generate realistic test images in-memory."""
    img = Image.new("RGB", (224, 224), color=color)
    draw = ImageDraw.Draw(img)
    if pattern == "circle":
        draw.ellipse([60, 60, 164, 164], fill=(255, 255, 255))
    elif pattern == "triangle":
        draw.polygon([(112, 35), (35, 185), (189, 185)], fill=(255, 255, 255))
    elif pattern == "rectangle":
        draw.rectangle([50, 50, 174, 174], fill=(200, 200, 200))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


# ==============================================================================
# TEST 1 & 4: NEW IMAGE & DIFFERENT IMAGE ANALYSIS
# ==============================================================================

def test_01_new_image_and_different_image():
    img1 = create_mock_image(color=(34, 197, 94), pattern="circle")
    img2 = create_mock_image(color=(220, 38, 38), pattern="triangle")

    analysis1 = analyze_comprehensive_image(img1)
    analysis2 = analyze_comprehensive_image(img2)

    assert analysis1["image"]["hash"] != analysis2["image"]["hash"]
    assert "description" in analysis1["image"]
    assert "perspectives" in analysis1
    assert len(analysis1["perspectives"]) >= 7


# ==============================================================================
# TEST 2 & 3: CACHE HIT & SAME IMAGE CHANGED SENTIMENT (<1ms)
# ==============================================================================

def test_02_same_image_cache_hit_and_instant_sentiment_change():
    img = create_mock_image(color=(59, 130, 246), pattern="rectangle")

    # Initial Analysis
    analysis = analyze_comprehensive_image(img)
    image_hash = analysis["image"]["hash"]

    # Match with Positive
    t0 = time.perf_counter()
    match_pos = calculate_sentiment_match(analysis, "Positive")
    t_pos_ms = (time.perf_counter() - t0) * 1000

    # Change to Negative - instant without re-running vision
    t1 = time.perf_counter()
    match_neg = calculate_sentiment_match(analysis, "Negative")
    t_neg_ms = (time.perf_counter() - t1) * 1000

    assert t_neg_ms < 5.0  # Sub-5ms execution guarantee (no vision re-run)
    assert match_pos["match_score"] != match_neg["match_score"]
    assert "explanation" in match_pos
    assert "explanation" in match_neg


# ==============================================================================
# TEST 5 & 6: CONFLICT DETECTION (POS/NEG & NEG/POS)
# ==============================================================================

def test_05_cross_modal_conflict_pos_text_neg_image():
    engine = SentiFusionInferenceEngine()
    neg_img = create_mock_image(color=(220, 38, 38), pattern="triangle")

    result = engine.predict("what a wonderful and glorious day", neg_img)
    assert result["cross_modal_conflict"]["has_conflict"] is True
    assert result["cross_modal_conflict"]["conflict_score"] >= 0.70
    assert result["sentiment"] == "Negative"  # Sarcasm resolution


def test_06_cross_modal_conflict_neg_text_pos_image():
    engine = SentiFusionInferenceEngine()
    pos_img = create_mock_image(color=(34, 197, 94), pattern="circle")

    result = engine.predict("Extremely terrible and dreadful failure", pos_img)
    assert result["cross_modal_conflict"]["has_conflict"] is True
    assert result["cross_modal_conflict"]["conflict_score"] >= 0.70


# ==============================================================================
# TEST 7, 8, 9, 10: FOUR MODALITY INFORMATIVENESS REGIMES
# ==============================================================================

def test_07_regime_image_dominant():
    engine = SentiFusionInferenceEngine()
    pos_img = create_mock_image(color=(34, 197, 94), pattern="circle")

    result = engine.predict("Routine package delivery on calendar schedule", pos_img)
    assert result["sentiment"] in ["Positive", "Neutral"]
    assert result["modality_weights"]["image_percentage"] >= 30.0


def test_08_regime_text_dominant():
    engine = SentiFusionInferenceEngine()
    neutral_img = create_mock_image(color=(100, 116, 139), pattern="rectangle")

    result = engine.predict("Completely awful, deeply unsatisfied and angry with service", neutral_img)
    assert result["sentiment"] == "Negative"
    assert result["modality_weights"]["text_percentage"] >= 40.0


def test_09_regime_both_informative_congruent():
    engine = SentiFusionInferenceEngine()
    pos_img = create_mock_image(color=(34, 197, 94), pattern="circle")

    result = engine.predict("Super happy, thrilled, and celebrating my dream milestone!", pos_img)
    assert result["sentiment"] == "Positive"
    assert result["confidence"] >= 65.0
    assert result["cross_modal_conflict"]["has_conflict"] is False


def test_10_regime_neither_informative_neutral():
    engine = SentiFusionInferenceEngine()
    gray_img = create_mock_image(color=(128, 128, 128), pattern="rectangle")

    result = engine.predict("Standard generic document page number 42", gray_img)
    assert result["sentiment"] == "Neutral"
    assert result["cross_modal_conflict"]["has_conflict"] is False


# ==============================================================================
# TEST 11: OCR & MULTIMODAL EXPLAINABILITY
# ==============================================================================

def test_11_explainability_and_faithfulness():
    engine = SentiFusionInferenceEngine()
    img = create_mock_image(color=(34, 197, 94), pattern="circle")

    result = engine.predict("Finally achieved my major graduation milestone", img, run_faithfulness=True)
    assert "explainability" in result
    assert "top_tokens" in result["explainability"]
    assert len(result["explainability"]["top_tokens"]) > 0
    assert "faithfulness" in result
    assert "faithfulness_score" in result["faithfulness"]


# ==============================================================================
# TEST 12: MODEL ABLATION SUITE (E1 to E8)
# ==============================================================================

def test_12_model_ablations_forward_pass():
    model = SentiFusionNet(vocab_size=100, num_classes=3)
    t = torch.randint(0, 100, (2, 16))
    img = torch.randn(2, 3, 224, 224)

    for mode in ["full", "concat", "cross_attn_only", "unidirectional", "global_features"]:
        out = model(t, img, ablation=mode)
        assert out["logits"].shape == (2, 3)
        assert out["conflict_prob"].shape == (2, 1)
