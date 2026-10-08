import os
import io
import time
import json
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageStat
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights
import torchvision.transforms as T

# Configure PyTorch CPU optimizations
torch.set_num_threads(4)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "analysis_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Try initializing OCR reader (EasyOCR / fallback)
_ocr_reader = None
try:
    import easyocr
    _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
except Exception as e:
    print(f"[Vision Pipeline] Notice on EasyOCR initialization: {e}")

# OpenCV Face Detection Haar Cascade
_face_cascade = None
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        _face_cascade = cv2.CascadeClassifier(cascade_path)
except Exception as e:
    print(f"[Vision Pipeline] Notice on Face Cascade: {e}")

# Pretrained ImageNet Model for Object/Scene Recognition
_imagenet_model = None
_imagenet_labels = None

def get_imagenet_model():
    global _imagenet_model, _imagenet_labels
    if _imagenet_model is None:
        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            _imagenet_model = models.mobilenet_v3_small(weights=weights).eval()
            _imagenet_labels = weights.meta["categories"]
        except Exception as e:
            print(f"[Vision Pipeline] Warning loading ImageNet backbone: {e}")
    return _imagenet_model, _imagenet_labels

# Image Transforms
_transform_pipeline = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global In-Memory Analysis Cache
_memory_cache: Dict[str, Dict[str, Any]] = {}


def compute_image_hash(image_bytes: bytes) -> str:
    """Computes SHA-256 hash of image bytes for robust deduplication & instant caching."""
    return hashlib.sha256(image_bytes).hexdigest()


def extract_dominant_colors(pil_img: Image.Image) -> Dict[str, Any]:
    """Extracts dominant color palette, brightness, contrast, and atmosphere."""
    img_rgb = pil_img.convert("RGB")
    stat = ImageStat.Stat(img_rgb)
    mean_r, mean_g, mean_b = stat.mean[:3]
    brightness = (mean_r * 299 + mean_g * 587 + mean_b * 114) / 1000

    # Color mood calculation
    is_alert_red = mean_r > 150 and mean_g < 100 and mean_b < 100
    is_cheerful_green = (mean_g > 120 and mean_r < 100) or (mean_g > mean_r + 25 and mean_g > mean_b + 25)
    is_sunny_warm = mean_r > 150 and mean_g > 130 and mean_b < 120
    is_cool = mean_b > mean_r + 25
    is_dark = brightness < 75
    is_neutral_gray = abs(mean_r - mean_g) < 20 and abs(mean_g - mean_b) < 20

    is_warm = is_sunny_warm or is_cheerful_green
    
    atmosphere = "Neutral / Balanced"
    if is_alert_red:
        atmosphere = "Hazard / Warning / Distress Tone"
    elif is_dark:
        atmosphere = "Dark / Somber / Moody"
    elif is_cheerful_green or is_sunny_warm:
        atmosphere = "Vibrant / Cheerful / Energetic"
    elif is_cool:
        atmosphere = "Cool / Calm / Serene"
    elif is_neutral_gray:
        atmosphere = "Neutral / Balanced"

    # Hex palette sampling (dominant 3 colors)
    small_img = img_rgb.resize((32, 32), Image.Resampling.LANCZOS)
    colors = small_img.getcolors(maxcolors=1024)
    palette = []
    if colors:
        sorted_colors = sorted(colors, key=lambda c: c[0], reverse=True)[:4]
        for _, rgb in sorted_colors:
            palette.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")

    return {
        "brightness": round(brightness, 1),
        "atmosphere": atmosphere,
        "is_warm": is_warm,
        "is_cool": is_cool,
        "palette": palette or [f"#{int(mean_r):02x}{int(mean_g):02x}{int(mean_b):02x}"]
    }


def detect_faces_and_people(cv_img_rgb: np.ndarray) -> Dict[str, Any]:
    """Detects faces, estimated people count, and facial emotional cues."""
    if _face_cascade is None:
        return {"people_count": "Unknown", "faces_detected": 0, "facial_expression": "Not enough visual evidence"}

    gray = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
    face_count = len(faces)

    if face_count == 0:
        return {
            "people_count": 0,
            "faces_detected": 0,
            "facial_expression": "No faces visible",
            "body_language": "Not enough visual evidence"
        }

    expressions = []
    for (x, y, w, h) in faces[:3]:
        roi_gray = gray[y:y+h, x:x+w]
        # Calculate smile/mouth region contrast
        lower_face = roi_gray[int(h*0.6):h, :]
        mean_intensity = np.mean(lower_face)
        std_intensity = np.std(lower_face)
        if std_intensity > 35:
            expressions.append("Dynamic / Expressive expression detected")
        else:
            expressions.append("Neutral / Calm facial expression")

    return {
        "people_count": face_count,
        "faces_detected": face_count,
        "facial_expression": ", ".join(set(expressions)),
        "body_language": f"{face_count} person(s) present with visible facial engagement"
    }


def extract_ocr_text(cv_img_rgb: np.ndarray, pil_img: Image.Image) -> Dict[str, Any]:
    """Extracts text visible inside the image using EasyOCR or fallback detection."""
    detected_lines = []
    avg_confidence = 0.0

    if _ocr_reader is not None:
        try:
            results = _ocr_reader.readtext(cv_img_rgb)
            for bbox, text, conf in results:
                cleaned = text.strip()
                if cleaned and len(cleaned) >= 2 and conf > 0.25:
                    detected_lines.append(cleaned)
                    avg_confidence += conf
            if detected_lines:
                avg_confidence = round(avg_confidence / len(detected_lines), 2)
        except Exception as e:
            print(f"[Vision Pipeline] EasyOCR error: {e}")

    full_text = " ".join(detected_lines)
    has_text = len(detected_lines) > 0

    return {
        "has_text": has_text,
        "lines": detected_lines,
        "full_text": full_text,
        "confidence": avg_confidence if has_text else 0.0
    }


def detect_objects_and_scene(pil_img: Image.Image) -> Dict[str, Any]:
    """Uses MobileNetV3 to extract recognized objects, entities, and visual concepts."""
    model, labels = get_imagenet_model()
    if model is None or labels is None:
        return {"objects": [], "scene": "Unknown", "entities": []}

    tensor = _transform_pipeline(pil_img).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0)

    top_indices = torch.topk(probs, k=5).indices.tolist()
    detected_objects = []
    for idx in top_indices:
        label = labels[idx].replace("_", " ")
        conf = float(probs[idx].item())
        if conf > 0.03:
            detected_objects.append({"label": label, "confidence": round(conf, 3)})

    # Infer broad scene
    labels_str = " ".join([o["label"] for o in detected_objects]).lower()
    scene = "General / Everyday Environment"
    if any(k in labels_str for k in ["car", "truck", "cab", "vehicle", "crash", "street", "traffic"]):
        scene = "Urban / Automotive / Road"
    elif any(k in labels_str for k in ["mountain", "lake", "forest", "tree", "valley", "alp"]):
        scene = "Natural Outdoor Landscape"
    elif any(k in labels_str for k in ["stage", "microphone", "guitar", "party", "restaurant", "dining"]):
        scene = "Social / Event / Indoor Venue"
    elif any(k in labels_str for k in ["office", "laptop", "desk", "computer", "screen"]):
        scene = "Workplace / Digital Office"

    return {
        "objects": detected_objects,
        "scene": scene,
        "top_labels": [o["label"] for o in detected_objects]
    }


def detect_damage_and_fractures(
    cv_img_rgb: np.ndarray,
    detected_objects: Optional[List[Dict[str, Any]]] = None,
    scene: str = "",
    people_count: int = 0
) -> Dict[str, Any]:
    """
    Detects glass fractures, cracked screens, spiderweb crack patterns, and physical damage
    only when visual structure, edge density, and context corroborate damage rather than
    general scene complexity (e.g. stage backdrops, clothing folds, text, or crowds).
    """
    try:
        gray = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        if h < 20 or w < 20:
            return {"has_damage": False, "damage_type": None, "crack_density": 0.0}

        obj_labels = " ".join([o.get("label", "") for o in (detected_objects or [])]).lower()
        
        # Check if scene/objects represent human social/celebration events
        celebration_or_human = (
            any(k in obj_labels for k in ["mortarboard", "academic gown", "gown", "stage", "groom", "suit", "dress", "party", "cheer", "flower", "balloon", "trophy", "diploma", "wedding", "ceremony"])
            or (isinstance(people_count, int) and people_count >= 1)
            or scene in ["Social / Event / Indoor Venue", "Natural Outdoor Landscape"]
        )
        
        # Hardware or vehicle context
        hardware_or_crash_context = any(
            k in obj_labels or k in scene.lower()
            for k in ["screen", "monitor", "television", "cellular telephone", "laptop", "handheld computer", "car", "cab", "truck", "crash", "collision", "windshield", "mirror", "wreck"]
        )

        blurred = cv2.bilateralFilter(gray, 7, 50, 50)
        edges = cv2.Canny(blurred, 40, 130)

        total_pixels = h * w
        edge_pixels = int(np.count_nonzero(edges))
        edge_density = edge_pixels / total_pixels

        # Detect sharp intersecting fracture lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=8)
        num_lines = len(lines) if lines is not None else 0

        angles = []
        if lines is not None and num_lines > 0:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                ang = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                angles.append(ang)

        angle_std = float(np.std(angles)) if len(angles) > 8 else 0.0

        if celebration_or_human and not hardware_or_crash_context:
            # Human/celebration scenes contain clothing folds, diplomas, stage curtains: do not misclassify as screen cracks
            is_fractured = False
        else:
            is_fractured = (edge_density > 0.03 and num_lines >= 20 and angle_std > 20.0) or (num_lines >= 45 and angle_std > 18.0)

        return {
            "has_damage": is_fractured,
            "damage_type": "Shattered / Cracked screen fracture" if is_fractured else None,
            "crack_density": round(edge_density * 100, 2),
            "fracture_lines": num_lines,
            "angle_variance": round(angle_std, 1)
        }
    except Exception as e:
        print(f"[Vision Pipeline] Notice on fracture detection: {e}")
        return {"has_damage": False, "damage_type": None, "crack_density": 0.0}


from .deep_reasoner import (
    classify_and_structure_entities,
    extract_actions_and_events,
    build_entity_relationship_graph,
    analyze_scene_observation_vs_interpretation,
    analyze_visual_emotions,
    analyze_implicit_meaning_and_symbolism
)


def analyze_comprehensive_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    Executes COMPLETE MULTI-PERSPECTIVE IMAGE UNDERSTANDING across all 12 perception levels.
    Checks memory/disk cache first. If found, returns in <1ms without re-running vision models.
    """
    image_hash = compute_image_hash(image_bytes)
    total_start = time.perf_counter()

    # 1. Check Memory Cache
    if image_hash in _memory_cache:
        cached = _memory_cache[image_hash].copy()
        cached["performance"] = {
            "cache_hit": True,
            "cache_type": "Memory Cache",
            "processing_time_ms": round((time.perf_counter() - total_start) * 1000, 2)
        }
        return cached

    # 2. Check Disk Cache
    disk_cache_file = os.path.join(CACHE_DIR, f"{image_hash}.json")
    if os.path.exists(disk_cache_file):
        try:
            with open(disk_cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            _memory_cache[image_hash] = cached
            cached["performance"] = {
                "cache_hit": True,
                "cache_type": "Disk Cache",
                "processing_time_ms": round((time.perf_counter() - total_start) * 1000, 2)
            }
            return cached
        except Exception:
            pass

    # 3. Process Image with Computer Vision Modules
    t_prep_start = time.perf_counter()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_img_rgb = np.array(pil_img)
    image_processing_time_ms = round((time.perf_counter() - t_prep_start) * 1000, 2)

    # A. Dominant Color & Atmosphere
    color_info = extract_dominant_colors(pil_img)

    # B. Face & People Detection
    people_info = detect_faces_and_people(cv_img_rgb)

    # C. OCR Text Extraction
    t_ocr_start = time.perf_counter()
    ocr_info = extract_ocr_text(cv_img_rgb, pil_img)
    ocr_time_ms = round((time.perf_counter() - t_ocr_start) * 1000, 2)

    # D. Object & Scene Recognition
    t_vision_start = time.perf_counter()
    object_info = detect_objects_and_scene(pil_img)
    vision_inference_time_ms = round((time.perf_counter() - t_vision_start) * 1000, 2)

    # E. Fracture / Damage & Defect Detection (Context-Aware)
    damage_info = detect_damage_and_fractures(
        cv_img_rgb,
        detected_objects=object_info["objects"],
        scene=object_info["scene"],
        people_count=people_info["people_count"] if isinstance(people_info["people_count"], int) else 0
    )

    # Contextual Celebration Cues
    dominant_objects = " ".join(object_info["top_labels"]).lower()
    text_content = ocr_info["full_text"].lower()
    has_celebration_cues = any(
        k in dominant_objects or k in text_content
        for k in ["celebration", "party", "congratulations", "win", "winner", "trophy", "stage", "flower", "gift", "smile", "happy", "fun", "love", "mortarboard", "academic gown", "gown", "diploma", "certificate", "graduation", "commencement", "achievement", "ceremony", "cheer"]
    )
    has_damage_cues = damage_info["has_damage"] or (
        not has_celebration_cues and any(k in dominant_objects or k in text_content for k in ["crash", "wreck", "accident", "damage", "broken", "danger", "warning", "distress", "police", "ambulance", "dent", "emergency"])
    )

    # F. Level 2 & 3: Deep Entity Classification (Living vs Non-Living)
    t_reason_start = time.perf_counter()
    all_entities, living_entities, non_living_entities = classify_and_structure_entities(
        detected_objects=object_info["objects"],
        people_count=people_info["people_count"] if isinstance(people_info["people_count"], int) else 0,
        damage_info=damage_info,
        ocr_lines=ocr_info["lines"]
    )

    # G. Level 4: Actions & Event Reasoning
    actions = extract_actions_and_events(
        scene=object_info["scene"],
        living_entities=living_entities,
        non_living_entities=non_living_entities,
        damage_info=damage_info,
        ocr_text=ocr_info["full_text"]
    )

    # H. Level 5: Entity Relationships Graph
    relationships = build_entity_relationship_graph(
        living_entities=living_entities,
        non_living_entities=non_living_entities,
        damage_info=damage_info,
        scene=object_info["scene"]
    )

    # I. Level 6: Scene Understanding (Observation vs Interpretation)
    scene_breakdown = analyze_scene_observation_vs_interpretation(
        scene_name=object_info["scene"],
        living_entities=living_entities,
        non_living_entities=non_living_entities,
        damage_info=damage_info,
        atmosphere=color_info["atmosphere"]
    )

    # J. Level 7: Visual Emotions
    visual_emotions = analyze_visual_emotions(
        faces_detected=people_info["faces_detected"],
        facial_expression=people_info["facial_expression"],
        damage_info=damage_info,
        atmosphere=color_info["atmosphere"],
        scene_name=object_info["scene"],
        has_celebration=has_celebration_cues
    )
    primary_emotion = visual_emotions[0]["emotion"] if visual_emotions else "Neutral"
    emotion_confidence = visual_emotions[0]["confidence"] if visual_emotions else 0.70
    emotion_evidence = [visual_emotions[0]["observable_evidence"]] if visual_emotions else ["Standard visual scene"]

    # K. Level 8, 9 & 10: Implicit Meaning, Symbolic Visual Cues, Meme/Irony
    explicit_content, symbolic_cues, implicit_meaning, possible_irony = analyze_implicit_meaning_and_symbolism(
        scene_name=object_info["scene"],
        damage_info=damage_info,
        ocr_text=ocr_info["full_text"],
        living_entities=living_entities,
        non_living_entities=non_living_entities,
        atmosphere=color_info["atmosphere"]
    )

    # L. Image Message & Intent
    message_intent = "Neutral Information"
    intent_confidence = 0.75
    if has_celebration_cues:
        message_intent = "Celebration / Achievement"
        intent_confidence = 0.92
    elif has_damage_cues:
        message_intent = "Warning / Incident / Hardware Defect"
        intent_confidence = 0.88
    elif ocr_info["has_text"] and any(k in text_content for k in ["sale", "buy", "discount", "offer", "price", "shop", "app"]):
        message_intent = "Advertisement / Promotion"
        intent_confidence = 0.85
    elif object_info["scene"] == "Natural Outdoor Landscape":
        message_intent = "Nature / Scenery Appreciation"
        intent_confidence = 0.82
    elif isinstance(people_info["people_count"], int) and people_info["people_count"] > 1:
        message_intent = "Social Interaction / Friendship"
        intent_confidence = 0.80

    image_message = {
        "message": message_intent,
        "confidence": intent_confidence,
        "evidence": [f"Scene identified as {object_info['scene']}", f"Atmosphere characterized as {color_info['atmosphere']}"]
    }

    # M. Polarity Calculation (Positive, Negative, Neutral)
    if has_celebration_cues or primary_emotion in ["Happiness / Engagement", "Uplift / Energy", "Happiness", "Excitement", "Love/affection", "Hope"]:
        pos_prob, neg_prob, neu_prob = 0.88, 0.04, 0.08
        dominant_sentiment = "Positive"
    elif primary_emotion in ["Frustration / Distress", "Somberness / Seriousness", "Hazard / Distress Tone", "Sadness", "Anger", "Fear", "Anxiety", "Fear / Concern"]:
        pos_prob, neg_prob, neu_prob = 0.05, 0.85, 0.10
        dominant_sentiment = "Negative"
    else:
        pos_prob, neg_prob, neu_prob = 0.15, 0.15, 0.70
        dominant_sentiment = "Neutral"

    # Element Breakdown
    positive_elements = []
    negative_elements = []
    neutral_elements = [object_info["scene"]]

    if has_celebration_cues or color_info["is_warm"]:
        positive_elements.append("Warm/vibrant visual atmosphere")
    if damage_info.get("has_damage"):
        negative_elements.append(f"Physical hardware defect: {damage_info.get('damage_type', 'Cracked screen / fractured glass')}")
    if has_damage_cues and not damage_info.get("has_damage"):
        negative_elements.append("Signs of distress, impact, or hazardous situation")
    if people_info["faces_detected"] > 0:
        neutral_elements.append(f"{people_info['people_count']} person(s) present")
    if ocr_info["has_text"]:
        neutral_elements.append(f"Embedded text: \"{ocr_info['full_text']}\"")
    if not positive_elements and not negative_elements:
        neutral_elements.append("Balanced visual tones without extreme polarity")

    # Natural Language Narrative Description
    people_desc = f"with {people_info['people_count']} person(s) visible" if people_info["people_count"] > 0 else "without people visible"
    text_desc = f"containing visible text: '{ocr_info['full_text']}'" if ocr_info["has_text"] else "with no prominent text"
    damage_desc = f"with visible {damage_info['damage_type'].lower()}" if damage_info.get("has_damage") else ""
    description = (
        f"The image depicts a {object_info['scene'].lower()} scene {people_desc} {damage_desc}, "
        f"featuring {', '.join([o['label'] for o in object_info['objects'][:3]]) or 'general scenery'}. "
        f"The visual atmosphere is {color_info['atmosphere'].lower()} {text_desc}. "
        f"The overall intent communicates {message_intent.lower()} evoking {primary_emotion.lower()}."
    ).replace("  ", " ")

    reasoning_time_ms = round((time.perf_counter() - t_reason_start) * 1000, 2)
    elapsed_ms = round((time.perf_counter() - total_start) * 1000, 2)

    # Uncertainty handling
    uncertainties = []
    if people_info["people_count"] == "Unknown":
        uncertainties.append("Human presence could not be conclusively determined.")
    if not object_info.get("objects"):
        uncertainties.append("Visual entities contain high ambiguity; interpretation is bounded by general scene composition.")

    analysis_result = {
        "image_analysis": {
            "scene": scene_breakdown,
            "entities": all_entities,
            "living_entities": living_entities,
            "non_living_entities": non_living_entities,
            "actions": actions,
            "relationships": relationships,
            "ocr": [{"text": line, "confidence": ocr_info["confidence"]} for line in ocr_info["lines"]],
            "visual_emotions": visual_emotions,
            "explicit_content": explicit_content,
            "implicit_meaning": implicit_meaning,
            "symbolic_cues": symbolic_cues,
            "possible_irony": possible_irony,
            "visual_sentiment": {
                "positive": pos_prob,
                "negative": neg_prob,
                "neutral": neu_prob,
                "dominant": dominant_sentiment,
                "confidence": round(max(pos_prob, neg_prob, neu_prob) * 100, 1)
            },
            "image_message": image_message,
            "uncertainties": uncertainties
        },
        "image": {
            "hash": image_hash,
            "description": description,
            "objects": object_info["objects"],
            "people_count": people_info["people_count"],
            "faces_detected": people_info["faces_detected"],
            "scene": object_info["scene"],
            "ocr_text": ocr_info["lines"],
            "full_ocr_text": ocr_info["full_text"],
            "atmosphere": color_info["atmosphere"],
            "palette": color_info["palette"],
            "brightness": color_info["brightness"]
        },
        "emotion_analysis": {
            "primary": primary_emotion,
            "confidence": emotion_confidence,
            "evidence": emotion_evidence,
            "facial_expression": people_info["facial_expression"],
            "body_language": people_info["body_language"]
        },
        "sentiment_analysis": {
            "positive": pos_prob,
            "negative": neg_prob,
            "neutral": neu_prob,
            "dominant": dominant_sentiment,
            "confidence": round(max(pos_prob, neg_prob, neu_prob) * 100, 1)
        },
        "message": {
            "interpretation": message_intent,
            "confidence": intent_confidence,
            "story": f"The image communicates a message of {message_intent.lower()} centered around {object_info['scene']}."
        },
        "perspectives": {
            "visual_perspective": f"Visible entities: {', '.join([e['entity'] for e in all_entities[:3]]) or 'Scenery'}; Atmosphere: {color_info['atmosphere']}.",
            "emotional_perspective": f"Primary emotion is {primary_emotion} with {int(emotion_confidence*100)}% confidence.",
            "contextual_perspective": f"Setting is {object_info['scene']} with {people_info['people_count']} subject(s).",
            "social_perspective": f"{people_info['people_count']} person(s) present ({people_info['facial_expression']}).",
            "textual_perspective": f"OCR Text: '{ocr_info['full_text'] or 'No readable text'}'." if ocr_info["has_text"] else "No visible text detected.",
            "semantic_perspective": f"Communicating {message_intent}.",
            "sentiment_perspective": f"Evaluated as {dominant_sentiment} ({int(max(pos_prob, neg_prob, neu_prob)*100)}% confidence)."
        },
        "elements": {
            "positive_elements": positive_elements,
            "negative_elements": negative_elements,
            "neutral_elements": neutral_elements,
            "ambiguous_elements": []
        },
        "performance": {
            "cache_hit": False,
            "cache_type": "Fresh Computation",
            "image_processing_time_ms": image_processing_time_ms,
            "vision_inference_time_ms": vision_inference_time_ms,
            "ocr_time_ms": ocr_time_ms,
            "reasoning_time_ms": reasoning_time_ms,
            "processing_time_ms": elapsed_ms
        },
        "uncertainty": uncertainties
    }

    # Save to memory and disk cache
    _memory_cache[image_hash] = analysis_result
    try:
        with open(disk_cache_file, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, indent=2)
    except Exception as e:
        print(f"[Vision Pipeline] Disk cache write notice: {e}")

    return analysis_result
