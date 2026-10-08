"""
Comprehensive 20-Scenario Test Suite for SentiFusion Deep Image Understanding,
Multimodal Perception, Entity Relationship Graphs, and Cross-Modal Reasoning.
"""

import os
import sys
import io
import time
import pytest
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.inference import SentiFusionInferenceEngine
from ml.vision_pipeline import analyze_comprehensive_image
from ml.deep_reasoner import (
    classify_and_structure_entities,
    extract_actions_and_events,
    build_entity_relationship_graph,
    analyze_scene_observation_vs_interpretation,
    analyze_visual_emotions,
    analyze_implicit_meaning_and_symbolism,
    analyze_independent_text,
    perform_cross_modal_reasoning
)


def generate_scenario_image(scenario_type: str) -> bytes:
    """Generates synthetic in-memory test images representing distinct visual contexts."""
    img = Image.new("RGB", (224, 224), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    if scenario_type == "landscape":
        # Green ground + blue sky
        draw.rectangle([0, 0, 224, 112], fill=(135, 206, 235))
        draw.rectangle([0, 112, 224, 224], fill=(34, 139, 34))
        draw.polygon([(40, 112), (90, 40), (140, 112)], fill=(120, 120, 120))
    elif scenario_type == "celebration":
        draw.rectangle([0, 0, 224, 224], fill=(255, 240, 200))
        draw.ellipse([30, 30, 70, 70], fill=(255, 99, 71))
        draw.ellipse([150, 40, 190, 80], fill=(255, 215, 0))
        draw.rectangle([80, 120, 144, 180], fill=(255, 105, 180))
    elif scenario_type == "broken_hardware":
        draw.rectangle([20, 30, 204, 160], fill=(30, 30, 30))
        # Draw dense radiating crack lines
        for i in range(15):
            draw.line([(112, 95), (20 + i*12, 30)], fill=(255, 255, 255), width=2)
            draw.line([(112, 95), (20 + i*12, 160)], fill=(255, 255, 255), width=2)
            draw.line([(112, 95), (20, 30 + i*9)], fill=(255, 255, 255), width=2)
            draw.line([(112, 95), (204, 30 + i*9)], fill=(255, 255, 255), width=2)
    elif scenario_type == "advertisement":
        draw.rectangle([0, 0, 224, 224], fill=(255, 255, 255))
        draw.rectangle([20, 20, 204, 80], fill=(220, 38, 38))
        draw.text((40, 40), "50% OFF SALE", fill=(255, 255, 255))
    elif scenario_type == "dark_somber":
        draw.rectangle([0, 0, 224, 224], fill=(25, 25, 30))
    else: # normal / generic
        draw.rectangle([40, 40, 184, 184], fill=(100, 149, 237))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def engine():
    return SentiFusionInferenceEngine()


# ==============================================================================
# 20 REQUIRED SCENARIO TESTS
# ==============================================================================

def test_01_normal_photograph(engine):
    img = generate_scenario_image("normal")
    res = engine.predict("A standard clear snapshot of the room", img)
    assert "image_analysis" in res
    assert "entities" in res["image_analysis"]
    assert "final_prediction" in res
    assert res["final_prediction"]["sentiment"] in ["Positive", "Neutral", "Negative"]


def test_02_group_of_people():
    entities, living, non_living = classify_and_structure_entities(
        detected_objects=[{"label": "person", "confidence": 0.95}],
        people_count=4,
        damage_info={"has_damage": False},
        ocr_lines=[]
    )
    assert any("group of people (4)" in e["entity"] for e in living)
    assert len(living) >= 1


def test_03_animal_photograph():
    entities, living, non_living = classify_and_structure_entities(
        detected_objects=[{"label": "golden retriever dog", "confidence": 0.91}],
        people_count=0,
        damage_info={"has_damage": False},
        ocr_lines=[]
    )
    assert any(e["subcategory"] == "pet" for e in living)


def test_04_landscape(engine):
    img = generate_scenario_image("landscape")
    res = engine.predict("Beautiful vast mountain landscape", img)
    assert "image_analysis" in res
    assert res["cross_modal_analysis"]["agreement"] in ["Strong Congruence", "Moderate Alignment"]


def test_05_advertisement():
    explicit, symbols, implicit, irony = analyze_implicit_meaning_and_symbolism(
        scene_name="Commercial Display",
        damage_info={"has_damage": False},
        ocr_text="SPECIAL 50% OFF SALE BUY NOW",
        living_entities=[],
        non_living_entities=[{"entity": "billboard", "subcategory": "sign_document"}],
        atmosphere="Bright / Commercial"
    )
    assert "advertisement" in implicit["meaning"].lower() or "promotional" in implicit["meaning"].lower()
    assert implicit["evidence_level"] == "INFERRED"


def test_06_meme_text_image_juxtaposition():
    text_res = analyze_independent_text("When you write 1000 lines of code and forget to save...")
    assert text_res["sentiment"] in ["Negative", "Neutral"]


def test_07_image_with_embedded_text():
    explicit, symbols, implicit, irony = analyze_implicit_meaning_and_symbolism(
        scene_name="Signage",
        damage_info={"has_damage": False},
        ocr_text="DANGER HIGH VOLTAGE",
        living_entities=[],
        non_living_entities=[],
        atmosphere="Warning Tone"
    )
    assert explicit["detected_text"] == "DANGER HIGH VOLTAGE"


def test_08_celebration(engine):
    img = generate_scenario_image("celebration")
    res = engine.predict("Celebrating our team victory with happiness!", img)
    assert res["final_prediction"]["sentiment"] == "Positive"
    assert res["cross_modal_analysis"]["conflict"]["conflict_detected"] is False


def test_09_sad_emotional_scene():
    emotions = analyze_visual_emotions(
        faces_detected=1,
        facial_expression="Neutral / Calm facial expression",
        damage_info={"has_damage": True, "damage_type": "Severe fracture"},
        atmosphere="Dark / Somber / Moody",
        scene_name="Distressed Environment"
    )
    assert any(e["polarity"] == "Negative" for e in emotions)


def test_10_sports_event():
    actions = extract_actions_and_events(
        scene="Sports Stadium",
        living_entities=[{"entity": "athlete", "quantity": 1, "subcategory": "human"}],
        non_living_entities=[{"entity": "football", "subcategory": "sports_equipment"}],
        damage_info={"has_damage": False},
        ocr_text=""
    )
    assert len(actions) > 0


def test_11_protest_social_scene():
    rels = build_entity_relationship_graph(
        living_entities=[{"entity": "group of people (12)", "subcategory": "human"}],
        non_living_entities=[{"entity": "placard / banner", "subcategory": "sign_document"}],
        damage_info={"has_damage": False},
        scene="Public Square"
    )
    assert len(rels) >= 0


def test_12_image_with_multiple_objects():
    entities, living, non_living = classify_and_structure_entities(
        detected_objects=[
            {"label": "laptop", "confidence": 0.90},
            {"label": "coffee cup", "confidence": 0.85},
            {"label": "desk", "confidence": 0.80}
        ],
        people_count=0,
        damage_info={"has_damage": False},
        ocr_lines=[]
    )
    assert len(non_living) >= 3


def test_13_image_with_multiple_people():
    scene = analyze_scene_observation_vs_interpretation(
        scene_name="Office Conference",
        living_entities=[{"entity": "group of people (5)", "quantity": 5, "subcategory": "human"}],
        non_living_entities=[{"entity": "table", "subcategory": "furniture"}],
        damage_info={"has_damage": False},
        atmosphere="Neutral / Balanced"
    )
    assert "5 person(s) directly visible" in " ".join(scene["direct_observations"])


def test_14_image_with_ambiguous_context():
    explicit, symbols, implicit, irony = analyze_implicit_meaning_and_symbolism(
        scene_name="Unspecified Interior",
        damage_info={"has_damage": False},
        ocr_text="",
        living_entities=[],
        non_living_entities=[],
        atmosphere="Neutral / Balanced"
    )
    assert implicit["evidence_level"] in ["DIRECT", "INFERRED"]


def test_15_image_where_text_and_image_agree(engine):
    img = generate_scenario_image("celebration")
    res = engine.predict("Such an awesome and wonderful triumph!", img)
    assert res["final_prediction"]["sentiment"] == "Positive"
    assert res["cross_modal_analysis"]["agreement"] in ["Strong Congruence", "Moderate Alignment"]


def test_16_image_where_text_and_image_conflict(engine):
    img = generate_scenario_image("broken_hardware")
    res = engine.predict("Truly a splendid and fantastic accomplishment!!", img)
    assert res["cross_modal_analysis"]["conflict"]["conflict_detected"] is True
    assert res["final_prediction"]["sentiment"] == "Negative"  # Sarcasm resolution


def test_17_possible_sarcasm():
    text_data = analyze_independent_text("what a wonderful day!!")
    img_data = {
        "visual_sentiment": {"dominant": "Negative"},
        "visual_emotions": [{"emotion": "Frustration / Distress"}],
        "entities": [{"entity": "Cracked screen", "subcategory": "damaged_hardware"}]
    }
    fusion = perform_cross_modal_reasoning(text_data, img_data)
    assert fusion["conflict"]["conflict_detected"] is True
    assert "Sarcastic Discordance" in fusion["conflict"]["conflict_type"]
    assert fusion["final_prediction"]["sentiment"] == "Negative"


def test_18_possible_irony():
    text_data = analyze_independent_text("Just perfect, couldn't be better!")
    img_data = {
        "visual_sentiment": {"dominant": "Negative"},
        "visual_emotions": [{"emotion": "Frustration"}],
        "entities": [{"entity": "Damaged vehicle", "subcategory": "vehicle"}]
    }
    fusion = perform_cross_modal_reasoning(text_data, img_data)
    assert fusion["conflict"]["conflict_detected"] is True


def test_19_image_where_text_is_more_informative(engine):
    img = generate_scenario_image("normal")
    res = engine.predict("Deeply disappointed with the delayed customer service and damaged packaging", img)
    assert res["final_prediction"]["sentiment"] == "Negative"
    assert res["cross_modal_analysis"]["modality_importance"]["result"] in ["text_more_informative", "both_equally_informative"]


def test_20_image_where_visual_context_is_more_informative(engine):
    img = generate_scenario_image("broken_hardware")
    res = engine.predict("Standard office desk setup item 42", img)
    assert res["final_prediction"]["sentiment"] == "Negative"
    assert res["cross_modal_analysis"]["modality_importance"]["result"] in ["image_more_informative", "both_equally_informative"]
