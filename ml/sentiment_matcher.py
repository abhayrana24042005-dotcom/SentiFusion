import time
from typing import Dict, Any, List

# Polarity and emotional mappings for nuanced sentiment comparison
SENTIMENT_TAXONOMY = {
    # Positive cluster
    "positive": {"polarity": "Positive", "valence": 1.0, "emotions": ["Happiness", "Excitement", "Love/affection", "Hope", "Joy"]},
    "happy": {"polarity": "Positive", "valence": 1.0, "emotions": ["Happiness", "Joy", "Excitement"]},
    "excited": {"polarity": "Positive", "valence": 0.9, "emotions": ["Excitement", "Happiness"]},
    "motivational": {"polarity": "Positive", "valence": 0.85, "emotions": ["Hope", "Excitement", "Achievement"]},
    "celebratory": {"polarity": "Positive", "valence": 0.95, "emotions": ["Happiness", "Excitement", "Joy"]},
    "hopeful": {"polarity": "Positive", "valence": 0.8, "emotions": ["Hope", "Calmness"]},
    "calm": {"polarity": "Positive", "valence": 0.6, "emotions": ["Calmness", "Neutral"]},

    # Negative cluster
    "negative": {"polarity": "Negative", "valence": -1.0, "emotions": ["Sadness", "Anger", "Fear", "Anxiety", "Fear / Concern", "Disgust"]},
    "sad": {"polarity": "Negative", "valence": -0.9, "emotions": ["Sadness", "Grief"]},
    "angry": {"polarity": "Negative", "valence": -0.95, "emotions": ["Anger", "Frustration"]},
    "fearful": {"polarity": "Negative", "valence": -0.85, "emotions": ["Fear", "Anxiety", "Fear / Concern"]},
    "anxious": {"polarity": "Negative", "valence": -0.75, "emotions": ["Anxiety", "Fear"]},

    # Neutral / Sarcastic cluster
    "neutral": {"polarity": "Neutral", "valence": 0.0, "emotions": ["Neutral", "Calmness"]},
    "sarcastic": {"polarity": "Negative", "valence": -0.6, "emotions": ["Conflict", "Confusion", "Anger", "Disgust"]}
}


def calculate_sentiment_match(
    image_analysis: Dict[str, Any],
    selected_sentiment: str,
    custom_text: str = ""
) -> Dict[str, Any]:
    """
    Ultra-Fast Sentiment Matching Engine (<1ms).
    Compares cached image understanding with user's target sentiment.
    """
    start_time = time.perf_counter()
    sel_clean = (selected_sentiment or "Neutral").strip().lower()

    # Default lookup or infer from text
    target_info = SENTIMENT_TAXONOMY.get(sel_clean, {
        "polarity": "Positive" if "good" in sel_clean or "great" in sel_clean or "happy" in sel_clean else ("Negative" if "bad" in sel_clean or "sad" in sel_clean or "angry" in sel_clean else "Neutral"),
        "valence": 1.0 if "good" in sel_clean else (-1.0 if "bad" in sel_clean else 0.0),
        "emotions": ["General"]
    })

    # Check if image_analysis has nested structure or top-level
    deep_img = image_analysis.get("image_analysis", image_analysis)
    
    detected_sentiment = deep_img.get("visual_sentiment", {}).get("dominant") or image_analysis.get("sentiment_analysis", {}).get("dominant", "Neutral")
    emotions_list = deep_img.get("visual_emotions", [])
    detected_emotion = emotions_list[0].get("emotion", "Neutral") if emotions_list else image_analysis.get("emotion_analysis", {}).get("primary", "Neutral")
    detected_intent = deep_img.get("image_message", {}).get("message") or image_analysis.get("message", {}).get("interpretation", "General")
    ocr_text = image_analysis.get("image", {}).get("full_ocr_text", "")
    uncertainties = list(deep_img.get("uncertainties", []))

    # Calculate Compatibility Match Score (0 - 100%)
    match_score = 50
    reasoning_points = []

    target_polarity = target_info["polarity"]
    if target_polarity == detected_sentiment:
        match_score = 92 if target_polarity != "Neutral" else 88
        reasoning_points.append(f"Strong agreement: User selected '{selected_sentiment}' which directly aligns with the detected '{detected_sentiment}' image polarity.")
    elif detected_sentiment == "Neutral":
        if target_polarity == "Positive":
            match_score = 64
            reasoning_points.append(f"Moderate positive compatibility with neutral visual background.")
        else:
            match_score = 42
            reasoning_points.append(f"Moderate negative divergence from neutral visual background.")
    elif target_polarity == "Neutral":
        match_score = 56
        reasoning_points.append(f"Moderate neutrality compatibility with {detected_sentiment.lower()} visual context.")
    else:
        match_score = 14
        reasoning_points.append(f"Significant polarity clash: User selected '{selected_sentiment}' ({target_polarity}), but image visual cues strongly express '{detected_sentiment}' sentiment.")

    # Emotion and Intent fine-tuning
    if detected_emotion in target_info.get("emotions", []):
        match_score = min(100, match_score + 8)
        reasoning_points.append(f"Visual emotion '{detected_emotion}' matches the emotional category of '{selected_sentiment}'.")

    # Sarcasm / Discordance detection if custom text provided
    is_sarcasm = False
    if custom_text:
        text_lower = custom_text.lower()
        if (any(w in text_lower for w in ["wonderful", "great", "awesome", "loved", "perfect", "amazing"]) and detected_sentiment == "Negative") or (target_polarity == "Positive" and detected_sentiment == "Negative"):
            is_sarcasm = True
            uncertainties.append("⚠️ Sarcasm / Irony Discordance: Positive textual language ('" + custom_text + "') is juxtaposed with negative visual reality. The true sentiment is Negative.")
            reasoning_points.append("Textual praise contrasts with visual damage/distress, indicating sarcastic irony.")

    # Match rating tier
    if match_score >= 80:
        match_level = "High Alignment"
    elif match_score >= 50:
        match_level = "Moderate / Partial Match"
    else:
        match_level = "Conflict / Low Match"

    # Narrative explanation
    explanation = " ".join(reasoning_points)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "selected_sentiment": selected_sentiment,
        "detected_sentiment": detected_sentiment,
        "detected_emotion": detected_emotion,
        "match_score": match_score,
        "match_level": match_level,
        "explanation": explanation,
        "reasoning": reasoning_points,
        "user_comparison": {
            "selected_sentiment": selected_sentiment,
            "target_polarity": target_polarity,
            "match_score": match_score,
            "explanation": explanation
        },
        "perspectives": {
            "user_sentiment_perspective": f"User target '{selected_sentiment}' has a {match_score}% compatibility score with the visual understanding ({match_level})."
        },
        "uncertainty": uncertainties,
        "performance": {
            "processing_time_ms": elapsed_ms,
            "cached_understanding_reused": True
        }
    }
