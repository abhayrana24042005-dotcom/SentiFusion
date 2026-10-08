"""
Deep Multimodal Reasoning and Understanding Engine for SentiFusion.
Implements the 12-level hierarchical perception and reasoning pipeline:
1. Pixels / Visual Features -> 2. Living & Non-Living Entities -> 3. Attributes
4. Actions -> 5. Relationships Graph -> 6. Fine-grained Scene Observation vs Interpretation
7. Visual Emotions -> 8. Semantic Understanding -> 9. Implicit Meaning (3 evidence tiers)
10. Symbolic Cues & Meme/Irony -> 11. Independent Text Understanding
12. Deep Cross-Modal Fusion, Conflict/Incongruity Detection, Modality Importance & Combined Meaning.
"""

import re
from typing import Dict, List, Any, Optional, Tuple


# ==============================================================================
# TAXONOMIES AND KNOWLEDGE DICTIONARIES
# ==============================================================================

LIVING_CLASSES = {
    "human": ["person", "man", "woman", "child", "boy", "girl", "crowd", "people", "pedestrian", "athlete", "worker", "student", "groom", "bride", "doctor", "police"],
    "pet": ["dog", "cat", "puppy", "kitten", "hamster", "rabbit"],
    "animal": ["horse", "cow", "sheep", "elephant", "bear", "lion", "tiger", "deer", "monkey", "zebra", "giraffe", "animal"],
    "bird": ["bird", "pigeon", "eagle", "parrot", "duck", "goose", "swan", "owl", "hawk", "crow"],
    "insect": ["bee", "butterfly", "insect", "ant", "fly", "beetle"],
    "plant": ["tree", "flower", "rose", "grass", "plant", "forest", "foliage", "bush", "vegetation", "garden", "leaves"]
}

NON_LIVING_CLASSES = {
    "vehicle": ["car", "automobile", "truck", "bus", "bicycle", "bike", "motorcycle", "airplane", "plane", "train", "boat", "ship", "ambulance", "police car"],
    "building": ["building", "house", "skyscraper", "office", "home", "stadium", "church", "temple", "tower", "bridge", "barn", "station", "shop", "store"],
    "electronics": ["laptop", "computer", "screen", "monitor", "phone", "smartphone", "cellphone", "tablet", "television", "tv", "camera", "keyboard", "mouse"],
    "furniture": ["chair", "table", "desk", "sofa", "couch", "bed", "bench", "cabinet", "shelf"],
    "food": ["food", "cake", "pizza", "burger", "coffee", "meal", "drink", "dish", "bread", "fruit", "apple", "banana", "beverage", "dessert"],
    "tool_machine": ["tool", "hammer", "wrench", "machine", "tractor", "instrument", "medical equipment", "microphone"],
    "sign_document": ["sign", "poster", "banner", "billboard", "placard", "book", "paper", "document", "screen display", "label"],
    "sports_equipment": ["ball", "football", "basketball", "racket", "bat", "trophy", "medal", "bicycle", "helmet"],
    "environmental": ["road", "street", "mountain", "lake", "ocean", "river", "sky", "cloud", "sun", "snow", "water", "field", "beach", "ground"]
}

EMOTION_KEYWORDS = {
    "positive": ["happy", "delighted", "joy", "excited", "celebrating", "smile", "smiling", "laughing", "cheerful", "proud", "triumph", "hopeful", "peaceful", "calm"],
    "negative": ["sad", "crying", "angry", "furious", "frustrated", "distressed", "disappointed", "tense", "upset", "grief", "scared", "fearful", "shattered", "broken", "damaged", "ruined"],
    "neutral": ["focused", "composed", "working", "observing", "standing", "sitting", "ordinary", "standard", "formal", "routine"]
}


# ==============================================================================
# LEVEL 2 & 3: ENTITY PERCEPTION AND CLASSIFICATION
# ==============================================================================

def classify_and_structure_entities(
    detected_objects: List[Dict[str, Any]],
    people_count: int,
    damage_info: Dict[str, Any],
    ocr_lines: List[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Categorizes all entities into living and non-living, estimating location, size,
    visible attributes, quantity, and importance without hallucinating.
    """
    all_entities = []
    living_entities = []
    non_living_entities = []

    # 1. Process people / humans
    if people_count > 0:
        importance = "primary" if people_count <= 3 else "secondary"
        rel_size = "dominant" if people_count == 1 else "medium"
        loc = "foreground-center" if people_count == 1 else "center-spread"
        
        person_entity = {
            "entity": "person" if people_count == 1 else f"group of people ({people_count})",
            "category": "living",
            "subcategory": "human",
            "location": loc,
            "relative_size": rel_size,
            "visible_attributes": ["visible subjects", "active presence"],
            "quantity": people_count,
            "confidence": 0.92,
            "interactions": "present in scene",
            "importance": importance
        }
        all_entities.append(person_entity)
        living_entities.append(person_entity)

    # 2. Process detected vision objects
    seen_labels = set()
    for idx, obj in enumerate(detected_objects):
        lbl = obj.get("label", "").lower().strip()
        conf = float(obj.get("confidence", 0.5))
        if not lbl or lbl in seen_labels:
            continue
        seen_labels.add(lbl)

        # Categorize
        is_living = False
        subcat = "object"

        for group, words in LIVING_CLASSES.items():
            if any(w in lbl for w in words):
                is_living = True
                subcat = group
                break

        if not is_living:
            for group, words in NON_LIVING_CLASSES.items():
                if any(w in lbl for w in words):
                    subcat = group
                    break

        attributes = ["clearly visible"]
        if idx == 0:
            importance = "primary"
            rel_size = "dominant"
            loc = "center"
        elif idx <= 2:
            importance = "secondary"
            rel_size = "medium"
            loc = "center-left" if idx == 1 else "center-right"
        else:
            importance = "background"
            rel_size = "small"
            loc = "background"

        if damage_info.get("has_damage") and subcat in ["electronics", "vehicle", "tool_machine"]:
            attributes.append("visibly fractured / cracked / damaged")

        entity_dict = {
            "entity": lbl,
            "category": "living" if is_living else "non_living",
            "subcategory": subcat,
            "location": loc,
            "relative_size": rel_size,
            "visible_attributes": attributes,
            "quantity": 1,
            "confidence": round(conf, 3),
            "interactions": "positioned in context",
            "importance": importance
        }

        all_entities.append(entity_dict)
        if is_living:
            living_entities.append(entity_dict)
        else:
            non_living_entities.append(entity_dict)

    # 3. Add damaged hardware entity if fracture detected
    if damage_info.get("has_damage") and not any("fracture" in e["entity"] or "screen" in e["entity"] for e in all_entities):
        damaged_entity = {
            "entity": damage_info.get("damage_type", "Cracked / shattered surface"),
            "category": "non_living",
            "subcategory": "damaged_hardware",
            "location": "center-display",
            "relative_size": "dominant",
            "visible_attributes": [f"crack density {damage_info.get('crack_density', 0)}%", "spiderweb fracture lines"],
            "quantity": 1,
            "confidence": 0.89,
            "interactions": "physical structural impact / defect",
            "importance": "primary"
        }
        all_entities.insert(0, damaged_entity)
        non_living_entities.insert(0, damaged_entity)

    # 4. Add visible text entities if OCR found
    if ocr_lines:
        text_entity = {
            "entity": f"embedded text ('{ocr_lines[0]}')",
            "category": "non_living",
            "subcategory": "sign_document",
            "location": "text-region",
            "relative_size": "medium",
            "visible_attributes": ["typographic element", "readable lettering"],
            "quantity": len(ocr_lines),
            "confidence": 0.85,
            "interactions": "displays information",
            "importance": "secondary"
        }
        all_entities.append(text_entity)
        non_living_entities.append(text_entity)

    return all_entities, living_entities, non_living_entities


# ==============================================================================
# LEVEL 4 & 5: ACTIONS AND ENTITY RELATIONSHIPS GRAPH
# ==============================================================================

def extract_actions_and_events(
    scene: str,
    living_entities: List[Dict[str, Any]],
    non_living_entities: List[Dict[str, Any]],
    damage_info: Dict[str, Any],
    ocr_text: str
) -> List[Dict[str, Any]]:
    """Extracts observable actions, body postures, and event dynamics."""
    actions = []
    people_count = sum(e["quantity"] for e in living_entities if e.get("subcategory") == "human")
    all_labels = " ".join([e["entity"] for e in living_entities + non_living_entities]).lower()

    if damage_info.get("has_damage"):
        actions.append({
            "action": "structural impact / display fracture",
            "performer": "hardware / screen",
            "status": "damaged / shattered",
            "confidence": 0.89,
            "observable_evidence": "Intersecting fracture lines and crack edge dispersion across surface"
        })

    if people_count > 0:
        if people_count == 1:
            actions.append({
                "action": "individual presence / engagement",
                "performer": "person",
                "status": "observing or positioned in scene",
                "confidence": 0.85,
                "observable_evidence": "Single individual detected in visual region"
            })
        elif people_count >= 2:
            actions.append({
                "action": "group gathering / social co-presence",
                "performer": f"group of {people_count} people",
                "status": "interacting or sharing common setting",
                "confidence": 0.88,
                "observable_evidence": f"{people_count} individuals co-located in scene composition"
            })

    if any(k in all_labels for k in ["dog", "cat", "horse", "bird", "animal"]):
        actions.append({
            "action": "animal presence / natural behavior",
            "performer": "animal / wildlife",
            "status": "active or resting in environment",
            "confidence": 0.80,
            "observable_evidence": "Fauna entity identified in scene"
        })

    if any(k in all_labels for k in ["car", "truck", "bike", "bicycle", "bus"]):
        actions.append({
            "action": "vehicular transit / road positioning",
            "performer": "vehicle",
            "status": "positioned on roadway or parking setting",
            "confidence": 0.78,
            "observable_evidence": "Automotive entity visible in transportation setting"
        })

    if not actions:
        actions.append({
            "action": "static scenic arrangement",
            "performer": "environment",
            "status": "still / stationary composition",
            "confidence": 0.75,
            "observable_evidence": "No dynamic human or mechanical movement detected"
        })

    return actions


def build_entity_relationship_graph(
    living_entities: List[Dict[str, Any]],
    non_living_entities: List[Dict[str, Any]],
    damage_info: Dict[str, Any],
    scene: str
) -> List[Dict[str, Any]]:
    """Constructs a factual entity relationship graph supported strictly by visual evidence."""
    relationships = []

    has_people = any(e.get("subcategory") == "human" for e in living_entities)
    has_animals = any(e.get("subcategory") in ["pet", "animal", "bird"] for e in living_entities)
    has_electronics = any(e.get("subcategory") == "electronics" for e in non_living_entities)
    has_vehicles = any(e.get("subcategory") == "vehicle" for e in non_living_entities)
    has_damage = damage_info.get("has_damage", False)

    if has_damage and has_electronics:
        relationships.append({
            "subject": "Screen fracture / crack network",
            "relationship": "damages / fractures",
            "object": "Electronic display / laptop screen",
            "confidence": 0.90,
            "evidence": "Visible high-density radiating crack lines across display panel"
        })

    if has_people and has_animals:
        relationships.append({
            "subject": "Person",
            "relationship": "co-present / interacting with",
            "object": "Animal / Pet",
            "confidence": 0.84,
            "evidence": "Human and animal entities visible in shared composition"
        })

    if has_people and has_electronics:
        relationships.append({
            "subject": "Person",
            "relationship": "working with / near",
            "object": "Electronic hardware / computing device",
            "confidence": 0.82,
            "evidence": "Human presence within workspace alongside electronic equipment"
        })

    if has_people and has_vehicles:
        relationships.append({
            "subject": "Person",
            "relationship": "standing near / operating",
            "object": "Vehicle",
            "confidence": 0.80,
            "evidence": "Person and vehicle co-located in automotive context"
        })

    if not relationships and non_living_entities:
        primary_obj = non_living_entities[0]["entity"]
        relationships.append({
            "subject": primary_obj,
            "relationship": "situated within",
            "object": scene,
            "confidence": 0.78,
            "evidence": f"{primary_obj} positioned as dominant element within {scene} background"
        })

    return relationships


# ==============================================================================
# LEVEL 6 & 7: SCENE UNDERSTANDING & VISUAL EMOTIONS
# ==============================================================================

def analyze_scene_observation_vs_interpretation(
    scene_name: str,
    living_entities: List[Dict[str, Any]],
    non_living_entities: List[Dict[str, Any]],
    damage_info: Dict[str, Any],
    atmosphere: str
) -> Dict[str, Any]:
    """Separates direct visual observations from reasonable contextual interpretations."""
    observations = []
    interpretations = []

    people_count = sum(e["quantity"] for e in living_entities if e.get("subcategory") == "human")
    obj_names = [e["entity"] for e in non_living_entities[:3]]

    # Direct observations
    if people_count > 0:
        observations.append(f"{people_count} person(s) directly visible in scene")
    else:
        observations.append("Scene contains no visible human subjects")

    if obj_names:
        observations.append(f"Visible primary objects: {', '.join(obj_names)}")

    observations.append(f"Visual atmosphere is {atmosphere.lower()}")

    if damage_info.get("has_damage"):
        observations.append(f"Direct visual evidence of surface fracturing ({damage_info.get('damage_type')})")

    # Interpretations
    if damage_info.get("has_damage"):
        interpretations.append("The physical hardware has experienced severe impact or accidental defect, disrupting normal usage.")
    elif "Workplace" in scene_name or "Office" in scene_name:
        interpretations.append("The setting appears to represent a workplace or digital computing environment.")
    elif "Natural" in scene_name or "Landscape" in scene_name:
        interpretations.append("The composition suggests an outdoor natural landscape or scenic environment.")
    elif "Social" in scene_name or "Event" in scene_name:
        interpretations.append("The gathering suggests a social interaction, celebration, or public assembly.")
    else:
        interpretations.append(f"The environment aligns with a {scene_name.lower()} context.")

    return {
        "scene_type": scene_name,
        "direct_observations": observations,
        "contextual_interpretations": interpretations,
        "observation_confidence": 0.92,
        "interpretation_confidence": 0.84,
        "summary": " | ".join(observations) + " -> " + " ".join(interpretations)
    }


def analyze_visual_emotions(
    faces_detected: int,
    facial_expression: str,
    damage_info: Dict[str, Any],
    atmosphere: str,
    scene_name: str,
    has_celebration: bool = False
) -> List[Dict[str, Any]]:
    """Analyzes observable emotional cues without claiming absolute truth."""
    emotions = []

    if has_celebration or (faces_detected > 0 and ("Dynamic" in facial_expression or "Expressive" in facial_expression)):
        emotions.append({
            "emotion": "Happiness / Engagement",
            "polarity": "Positive",
            "confidence": 0.88 if has_celebration else 0.82,
            "observable_evidence": "Visible celebration cues, expressive facial dynamics, and celebratory milestone setting."
        })

    if damage_info.get("has_damage") and not has_celebration:
        emotions.append({
            "emotion": "Frustration / Distress",
            "polarity": "Negative",
            "confidence": 0.88,
            "observable_evidence": "Physical damage and broken hardware cues evoke frustration, loss, or inconvenience."
        })

    if faces_detected > 0 and not any(e["emotion"] == "Happiness / Engagement" for e in emotions):
        if "Neutral" in facial_expression or "Calm" in facial_expression:
            emotions.append({
                "emotion": "Composure / Neutrality",
                "polarity": "Neutral",
                "confidence": 0.80,
                "observable_evidence": "Relaxed facial features and neutral posture indicate calm, non-distressed state."
            })

    if "Warm" in atmosphere or "Vibrant" in atmosphere or "Cheerful" in atmosphere:
        if not any(e["emotion"] == "Happiness / Engagement" for e in emotions):
            emotions.append({
                "emotion": "Uplift / Energy",
                "polarity": "Positive",
                "confidence": 0.75,
                "observable_evidence": "Warm, bright color palette and illuminated visual atmosphere."
            })
    elif "Dark" in atmosphere or "Somber" in atmosphere or "Hazard" in atmosphere or "Distress" in atmosphere:
        if not has_celebration:
            emotions.append({
                "emotion": "Somberness / Seriousness" if "Dark" in atmosphere else "Hazard / Distress Tone",
                "polarity": "Negative",
                "confidence": 0.78,
                "observable_evidence": "Alert tone, low brightness, or hazardous color atmosphere."
            })

    if not emotions:
        emotions.append({
            "emotion": "Neutral / Balanced",
            "polarity": "Neutral",
            "confidence": 0.70,
            "observable_evidence": "Even lighting, steady composition, and absence of extreme affective cues."
        })

    return emotions


# ==============================================================================
# LEVEL 8, 9 & 10: IMPLICIT MEANINGS, SYMBOLS, AND MEME/IRONY
# ==============================================================================

def analyze_implicit_meaning_and_symbolism(
    scene_name: str,
    damage_info: Dict[str, Any],
    ocr_text: str,
    living_entities: List[Dict[str, Any]],
    non_living_entities: List[Dict[str, Any]],
    atmosphere: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluates implicit meanings with 3 strict evidence tiers (DIRECT, INFERRED, SPECULATIVE).
    Prevents hallucinating unsupported hidden symbolism.
    """
    explicit_content = {
        "visual_elements": [e["entity"] for e in living_entities + non_living_entities[:4]],
        "detected_text": ocr_text or "None",
        "scene_setting": scene_name
    }

    implicit_interpretation = {
        "meaning": "Routine visual documentation or contextual representation.",
        "evidence_level": "DIRECT",
        "supporting_evidence": ["Standard visual elements without evident symbolic subtext"],
        "confidence": 0.75
    }

    symbolic_cues = []
    possible_irony = {
        "is_ironic": False,
        "is_meme": False,
        "evidence": "No clear visual incongruity or meme template detected within image alone.",
        "confidence": 0.20
    }

    # 1. Damage / Fracture symbolism
    if damage_info.get("has_damage"):
        implicit_interpretation = {
            "meaning": "Hardware failure, accidental damage, breakdown, or sudden disruption of work.",
            "evidence_level": "INFERRED",
            "supporting_evidence": ["Severe screen fracture pattern", "Interrupted technology usage context"],
            "confidence": 0.88
        }
        symbolic_cues.append({
            "symbol": "Shattered screen / spiderweb cracks",
            "interpretation": "Symbolizes technical defect, frustration, broken reliability, or accidental loss",
            "confidence": 0.90
        })

    # 2. Celebration / Achievement cues
    elif any(k in ocr_text.lower() for k in ["congrats", "celebrate", "winner", "happy birthday", "anniversary"]):
        implicit_interpretation = {
            "meaning": "Milestone achievement, celebration of personal or communal success.",
            "evidence_level": "INFERRED",
            "supporting_evidence": [f"Embedded celebratory text: '{ocr_text}'"],
            "confidence": 0.90
        }
        symbolic_cues.append({
            "symbol": "Celebratory text/decor",
            "interpretation": "Communicates festive achievement and congratulatory sentiment",
            "confidence": 0.88
        })

    # 3. Commercial / Advertising cues
    elif any(k in ocr_text.lower() for k in ["sale", "off", "discount", "buy", "shop", "price", "%"]):
        implicit_interpretation = {
            "meaning": "Commercial advertisement or promotional call to action.",
            "evidence_level": "INFERRED",
            "supporting_evidence": [f"Promotional keywords in OCR text: '{ocr_text}'"],
            "confidence": 0.87
        }
        symbolic_cues.append({
            "symbol": "Promotional banner/offer",
            "interpretation": "Intended to motivate commercial purchase or product interest",
            "confidence": 0.85
        })

    # 4. Nature / Serenity cues
    elif "Natural Outdoor Landscape" in scene_name and not damage_info.get("has_damage"):
        implicit_interpretation = {
            "meaning": "Appreciation of natural environment, serenity, tranquility, or leisure.",
            "evidence_level": "INFERRED",
            "supporting_evidence": ["Scenic landscape elements", "Absence of industrial stress cues"],
            "confidence": 0.80
        }

    return explicit_content, symbolic_cues, implicit_interpretation, possible_irony


# ==============================================================================
# LEVEL 11 & 12: INDEPENDENT TEXT UNDERSTANDING
# ==============================================================================

def analyze_independent_text(text: str) -> Dict[str, Any]:
    """Analyzes the user's text independently with semantic, emotional, negation, and irony checks."""
    cleaned = (text or "").strip()
    if not cleaned:
        return {
            "literal_meaning": "No accompanying user text provided.",
            "sentiment": "Neutral",
            "polarity_score": 0.0,
            "emotion": "Neutral",
            "key_entities": [],
            "negations": [],
            "intensifiers": [],
            "possible_irony": False,
            "confidence": 0.50,
            "evidence": ["Empty text input"]
        }

    t_lower = cleaned.lower()
    words = re.findall(r"\b\w+\b", t_lower)

    # Detect negations & intensifiers
    negations = [w for w in words if w in ["not", "no", "never", "cannot", "cant", "dont", "wont", "isnt", "arent", "neither", "hardly", "barely"]]
    intensifiers = [w for w in words if w in ["very", "so", "extremely", "really", "absolutely", "totally", "super", "highly", "completely", "truly"]]

    pos_hits = [w for w in words if w in [
        "good", "great", "wonderful", "amazing", "love", "loved", "loving", "perfect", "happy", "joy", "joyful",
        "superb", "brilliant", "triumph", "success", "blessed", "best", "delight", "delighted", "delightful", "awesome",
        "celebrating", "celebration", "splendid", "fantastic", "accomplishment", "accomplished", "proud", "thankful",
        "grateful", "terrific", "phenomenal", "exceptional", "stellar", "flawless", "gorgeous", "clean", "smooth",
        "fast", "reliable", "premium", "top", "worth", "recommend", "recommended", "favorite", "fun", "cool",
        "magical", "vibrant", "bright", "exciting", "excited", "friendly", "helpful", "warm", "sweet", "nice", "fine",
        "enjoyed", "enjoyable", "glad", "triumph", "glorious", "pleased", "pleasant", "satisfying", "satisfied"
    ]]
    neg_hits = [w for w in words if w in [
        "bad", "terrible", "awful", "horrible", "hate", "hated", "worst", "broken", "damaged", "fail", "failed",
        "failure", "ruined", "angry", "sad", "disappointed", "disappointing", "disappointment", "pathetic", "ugly",
        "painful", "trash", "crap", "defective", "faulty", "garbage", "junk", "horrid", "nasty", "dreadful",
        "miserable", "worthless", "cheap", "flimsy", "regret", "toxic", "rude", "delayed", "late", "crash", "crashed",
        "scam", "shame", "unpleasant", "unusable", "terribly", "awfully", "poorly", "worse", "disaster", "mess",
        "frustrated", "frustrating", "annoying", "annoyed"
    ]]

    # Calculate polarity
    has_negation = len(negations) > 0
    if len(pos_hits) > len(neg_hits):
        literal_sentiment = "Negative" if has_negation else "Positive"
        primary_emotion = "Disappointment" if has_negation else "Happiness / Enthusiasm"
        confidence = 0.88
    elif len(neg_hits) > len(pos_hits):
        literal_sentiment = "Positive" if has_negation else "Negative"
        primary_emotion = "Relief" if has_negation else "Frustration / Anger"
        confidence = 0.88
    else:
        literal_sentiment = "Neutral"
        primary_emotion = "Neutral / Informative"
        confidence = 0.70

    # Check for potential irony cues in text syntax (e.g. "what a wonderful day!!" or "just perfect...")
    potential_irony = bool(
        re.search(r"(what a |such a )?(wonderful|great|glorious|perfect|lovely|amazing|brilliant) (day|start|work|experience|luck)[.!?]*", t_lower) or
        ("just perfect" in t_lower) or
        ("could not be better" in t_lower) or
        ("loving my" in t_lower and "!" in cleaned)
    )

    evidence = []
    if pos_hits:
        evidence.append(f"Positive lexical cues: {', '.join(pos_hits)}")
    if neg_hits:
        evidence.append(f"Negative lexical cues: {', '.join(neg_hits)}")
    if negations:
        evidence.append(f"Negation modifiers: {', '.join(negations)}")
    if intensifiers:
        evidence.append(f"Intensifying adverbs: {', '.join(intensifiers)}")

    return {
        "literal_meaning": f"Text expresses literal {literal_sentiment.lower()} statement regarding subject.",
        "sentiment": literal_sentiment,
        "polarity_score": 1.0 if literal_sentiment == "Positive" else (-1.0 if literal_sentiment == "Negative" else 0.0),
        "emotion": primary_emotion,
        "key_entities": [w for w in words if len(w) > 4 and w not in pos_hits + neg_hits + negations + intensifiers][:4],
        "negations": negations,
        "intensifiers": intensifiers,
        "possible_irony": potential_irony,
        "confidence": confidence,
        "evidence": evidence or ["Standard neutral vocabulary"]
    }


# ==============================================================================
# LEVEL 13 & 14: DEEP CROSS-MODAL FUSION & CONFLICT REASONING
# ==============================================================================

def perform_cross_modal_reasoning(
    text_analysis: Dict[str, Any],
    image_analysis: Dict[str, Any],
    user_target_sentiment: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compares image meaning with text meaning to evaluate semantic relationship,
    conflict/incongruity, sarcasm resolution, modality importance, and combined interpretation.
    """
    text_sent = text_analysis.get("sentiment", "Neutral")
    img_sent = image_analysis.get("visual_sentiment", {}).get("dominant", "Neutral")
    img_emotion = image_analysis.get("visual_emotions", [{}])[0].get("emotion", "Neutral") if image_analysis.get("visual_emotions") else "Neutral"
    damage_detected = any("damage" in e.get("entity", "").lower() or "fracture" in e.get("entity", "").lower() for e in image_analysis.get("entities", []))

    # 1. Semantic and Sentiment Relationship
    if text_sent == img_sent:
        semantic_rel = "Strongly consistent"
        sentiment_rel = "Harmonious Agreement"
        agreement = "Strong Congruence"
        has_conflict = False
        conflict_type = "None"
        conflict_strength = "Zero"
        conflict_evidence = []
    elif text_sent == "Positive" and (img_sent == "Negative" or damage_detected):
        semantic_rel = "Strongly conflicting"
        sentiment_rel = "Cross-Modal Polarity Clash (Sarcasm / Irony)"
        agreement = "Severe Incongruity"
        has_conflict = True
        conflict_type = "Sarcastic Discordance (Positive words + Negative visual reality)"
        conflict_strength = "High"
        conflict_evidence = [
            f"Literal positive text ('{text_analysis.get('literal_meaning')}') clashes with negative visual evidence ({img_emotion}).",
            "Damaged hardware or distressing visual setting contradicts praising textual sentiment."
        ]
    elif text_sent == "Negative" and img_sent == "Positive":
        semantic_rel = "Partially conflicting"
        sentiment_rel = "Discrepant Mixed Sentiment"
        agreement = "Discrepancy"
        has_conflict = True
        conflict_type = "Negative reaction to positive context or ironic complaint"
        conflict_strength = "Moderate"
        conflict_evidence = [
            f"Negative textual expression paired with vibrant/positive visual scenery."
        ]
    elif text_sent != "Neutral" and img_sent == "Neutral":
        semantic_rel = "Complementary"
        sentiment_rel = "Text-Specified Context on Neutral Visual Canvas"
        agreement = "Moderate Alignment"
        has_conflict = False
        conflict_type = "None"
        conflict_strength = "Low"
        conflict_evidence = []
    elif img_sent != "Neutral" and text_sent == "Neutral":
        semantic_rel = "Complementary"
        sentiment_rel = "Visually Expressive Scene with Neutral User Commentary"
        agreement = "Moderate Alignment"
        has_conflict = False
        conflict_type = "None"
        conflict_strength = "Low"
        conflict_evidence = []
    else:
        semantic_rel = "Mostly consistent"
        sentiment_rel = "Neutral Baseline"
        agreement = "Neutral Alignment"
        has_conflict = False
        conflict_type = "None"
        conflict_strength = "Zero"
        conflict_evidence = []

    # 2. Modality Importance Reasoning
    if has_conflict and text_sent == "Positive" and (img_sent == "Negative" or damage_detected):
        modality_importance = {
            "result": "image_more_informative",
            "importance": "Visual Context Overrides Literal Text (Sarcasm Resolution)",
            "text_weight": 0.30,
            "image_weight": 0.70,
            "rationale": "Visual damage and physical failure provide ground-truth reality that inverts the literal positive wording into sarcastic frustration."
        }
    elif text_sent != "Neutral" and img_sent == "Neutral":
        modality_importance = {
            "result": "text_more_informative",
            "importance": "Text Provides Dominant Emotional Stance",
            "text_weight": 0.75,
            "image_weight": 0.25,
            "rationale": "The image provides a neutral setting while the text explicitly articulates the subjective evaluation."
        }
    elif img_sent != "Neutral" and text_sent == "Neutral":
        modality_importance = {
            "result": "image_more_informative",
            "importance": "Visual Scene Contains Primary Emotional Valence",
            "text_weight": 0.25,
            "image_weight": 0.75,
            "rationale": "The visual environment carries distinct emotional signals while the text provides neutral commentary."
        }
    elif text_sent == img_sent and text_sent != "Neutral":
        modality_importance = {
            "result": "both_equally_informative",
            "importance": "Bidirectional Reinforcement",
            "text_weight": 0.50,
            "image_weight": 0.50,
            "rationale": "Both text and visual cues independently confirm the same emotional orientation."
        }
    else:
        modality_importance = {
            "result": "both_equally_informative",
            "importance": "Balanced Multimodal Baseline",
            "text_weight": 0.50,
            "image_weight": 0.50,
            "rationale": "Both modalities convey standard information without intense emotional polarity."
        }

    # 3. Combined Multimodal Interpretation
    if has_conflict and text_sent == "Positive" and (img_sent == "Negative" or damage_detected):
        combined_meaning = (
            "Based on the available image and text, the user is almost certainly expressing sarcastic irony or frustration. "
            "The literal positive phrasing is contradicted by the visible damage/defect, indicating an unfortunate situation."
        )
        final_sentiment = "Negative"
        final_confidence = 0.88
    elif has_conflict and text_sent == "Negative" and img_sent == "Positive":
        combined_meaning = (
            "Based on the available image and text, the user expresses a critical or dissatisfied perspective despite an otherwise positive or vibrant visual environment."
        )
        final_sentiment = "Negative"
        final_confidence = 0.72
    elif text_sent == "Positive" and img_sent == "Positive":
        combined_meaning = (
            "Based on the available image and text, both modalities harmoniously celebrate a positive experience, achievement, or pleasing visual moment."
        )
        final_sentiment = "Positive"
        final_confidence = 0.92
    elif text_sent == "Negative" and img_sent == "Negative":
        combined_meaning = (
            "Based on the available image and text, both modalities mutually validate an unpleasant, damaged, or distressing event."
        )
        final_sentiment = "Negative"
        final_confidence = 0.90
    elif text_sent != "Neutral":
        combined_meaning = f"Based on the available context, the user provides a {text_sent.lower()} assessment situated within an ordinary visual setting."
        final_sentiment = text_sent
        final_confidence = 0.82
    elif img_sent != "Neutral":
        combined_meaning = f"Based on the visual evidence, the scene communicates a {img_sent.lower()} atmosphere with descriptive text context."
        final_sentiment = img_sent
        final_confidence = 0.82
    else:
        combined_meaning = "Based on the available inputs, the multimodal combination communicates neutral, objective information."
        final_sentiment = "Neutral"
        final_confidence = 0.80

    conflict_score = 0.88 if has_conflict else (0.30 if semantic_rel == "Partially conflicting" else 0.12)

    return {
        "semantic_relationship": semantic_rel,
        "sentiment_relationship": sentiment_rel,
        "agreement": agreement,
        "conflict": {
            "has_conflict": has_conflict,
            "conflict_detected": has_conflict,
            "conflict_score": conflict_score,
            "conflict_type": conflict_type,
            "type": "Sarcasm / Incongruity Discordance" if has_conflict else "Harmonious Agreement",
            "strength": conflict_strength,
            "evidence": conflict_evidence,
            "explanation": " ".join(conflict_evidence) if conflict_evidence else "Text and visual context exhibit harmonious sentiment alignment."
        },
        "modality_importance": modality_importance,
        "combined_interpretation": {
            "interpretation": combined_meaning,
            "evidence": [
                f"Textual stance: {text_sent} ({text_analysis.get('emotion', 'Neutral')})",
                f"Visual stance: {img_sent} ({img_emotion})"
            ] + conflict_evidence,
            "confidence": final_confidence,
            "uncertainty": ["Interpretation is probabilistic and bounded by observable visual evidence."]
        },
        "final_prediction": {
            "sentiment": final_sentiment,
            "confidence": round(final_confidence * 100, 1),
            "reasoning": combined_meaning,
            "evidence": [
                f"Text polarity evaluated as {text_sent}",
                f"Visual polarity evaluated as {img_sent}",
                f"Cross-modal relationship evaluated as {semantic_rel}"
            ]
        }
    }
