import io
import sys
import os
import time
from typing import Optional, Dict, Any
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, Form, File, UploadFile, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure root project path is on sys.path for ml imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.inference import get_inference_engine
from ml.vision_pipeline import analyze_comprehensive_image, _memory_cache, CACHE_DIR
from ml.sentiment_matcher import calculate_sentiment_match
from ml.feedback_tracker import log_analysis_feedback, evaluate_feedback_metrics

# Initialize FastAPI application
app = FastAPI(
    title="SentiFusion Intelligent Multimodal API",
    description="Multimodal Image Sentiment Understanding, Multi-Perspective Analysis & Instant Cache-Assisted Sentiment Matching",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@app.on_event("startup")
def startup_event():
    """Pre-warm ML models during server boot to eliminate cold-start latency."""
    print("[SentiFusion Backend] Warming up vision and multimodal inference pipelines...")
    try:
        get_inference_engine()
        print("[SentiFusion Backend] Multimodal AI pipelines initialized and ready.")
    except Exception as e:
        print(f"[SentiFusion Backend] Notice warming up engine: {e}")


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend service readiness and ML status."""
    try:
        engine = get_inference_engine()
        has_model = engine is not None and hasattr(engine, "model") and engine.model is not None
        is_checkpoint_active = getattr(engine, "is_checkpoint_loaded", False)
        vocab_size = getattr(engine.tokenizer, "vocab_size", 0) if hasattr(engine, "tokenizer") else 0
    except Exception:
        has_model = False
        is_checkpoint_active = False
        vocab_size = 0

    return {
        "status": "healthy" if has_model else "degraded",
        "service": "SentiFusion Backend",
        "version": "3.0.0",
        "model_loaded": has_model,
        "checkpoint_active": is_checkpoint_active,
        "vocab_size": vocab_size,
        "cached_images_in_memory": len(_memory_cache),
        "pipeline_ready": has_model
    }


@app.post("/analyze-image")
async def analyze_image_endpoint(image: UploadFile = File(...)):
    """
    Complete Multi-Perspective Image Understanding Endpoint (Cached by SHA-256).
    Extracts objects, people, facial expressions, emotions, OCR text, atmosphere, and 8 perspectives.
    """
    if image is None or not image.filename:
        return JSONResponse(status_code=400, content={"success": False, "message": "Image file is required."})

    try:
        contents = await image.read()
        if len(contents) == 0:
            return JSONResponse(status_code=400, content={"success": False, "message": "Image is empty."})

        # Validate with Pillow
        with Image.open(io.BytesIO(contents)) as img:
            img.verify()

        analysis_result = analyze_comprehensive_image(contents)
        return {"success": True, "data": analysis_result}

    except Exception as e:
        print(f"[Backend Error /analyze-image] {e}")
        return JSONResponse(status_code=500, content={"success": False, "message": f"Image analysis error: {str(e)}"})


@app.post("/match-sentiment")
async def match_sentiment_endpoint(
    image_hash: str = Form(...),
    selected_sentiment: str = Form(...),
    custom_text: Optional[str] = Form("")
):
    """
    Fast Sentiment Matching Endpoint (<1ms).
    Reuses existing cached image analysis without re-running vision or OCR models.
    """
    # 1. Retrieve cached analysis
    image_analysis = _memory_cache.get(image_hash)
    if not image_analysis:
        # Check disk cache
        disk_file = os.path.join(CACHE_DIR, f"{image_hash}.json")
        if os.path.exists(disk_file):
            import json
            with open(disk_file, "r", encoding="utf-8") as f:
                image_analysis = json.load(f)
            _memory_cache[image_hash] = image_analysis

    if not image_analysis:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": f"No cached analysis found for image hash: {image_hash}. Please analyze image first."}
        )

    # 2. Run fast sentiment matching
    match_result = calculate_sentiment_match(
        image_analysis=image_analysis,
        selected_sentiment=selected_sentiment,
        custom_text=custom_text or ""
    )

    return {
        "success": True,
        "image_hash": image_hash,
        "match": match_result
    }


@app.get("/analysis/{image_hash}")
def get_analysis_by_hash(image_hash: str):
    """Retrieves cached image analysis by SHA-256 hash."""
    cached = _memory_cache.get(image_hash)
    if cached:
        return {"success": True, "cached": True, "data": cached}

    disk_file = os.path.join(CACHE_DIR, f"{image_hash}.json")
    if os.path.exists(disk_file):
        import json
        with open(disk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        _memory_cache[image_hash] = data
        return {"success": True, "cached": True, "data": data}

    raise HTTPException(status_code=404, detail="Image analysis not found in cache.")


@app.post("/analyze")
async def analyze_unified(
    text: Optional[str] = Form(None),
    selected_sentiment: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Unified Multimodal Analysis Endpoint with Instant Cache Reuse.
    1. If image is new: executes full multi-perspective understanding + OCR and caches it.
    2. If image is cached: retrieves analysis in <0.5ms.
    3. Runs deep neural multimodal inference + sentiment matching.
    """
    target_text = (text or "").strip()
    target_sentiment = (selected_sentiment or "Neutral").strip()

    if image is None or not image.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "message": "Image is required."}
        )

    try:
        contents = await image.read()
        if len(contents) == 0:
            return JSONResponse(status_code=400, content={"success": False, "message": "Uploaded image file is empty."})

        with Image.open(io.BytesIO(contents)) as img:
            img.verify()

    except (UnidentifiedImageError, Exception):
        return JSONResponse(status_code=400, content={"success": False, "message": "Invalid or corrupted image file."})

    try:
        # Step 1: Comprehensive Image Understanding (with SHA-256 Cache)
        image_understanding = analyze_comprehensive_image(contents)

        # Step 2: Fast Sentiment Matching & Discrepancy Detection
        sentiment_match = calculate_sentiment_match(
            image_analysis=image_understanding,
            selected_sentiment=target_sentiment,
            custom_text=target_text
        )

        # Step 3: Neural Multimodal Inference
        engine = get_inference_engine()
        neural_prediction = engine.predict(text=target_text, image_bytes=contents)

        # Merge insights into cohesive response
        combined_prediction = {
            **neural_prediction,
            "image_understanding": image_understanding,
            "sentiment_match": sentiment_match,
            "perspectives": image_understanding.get("perspectives", {})
        }

        return {
            "success": True,
            "message": "Multimodal sentiment and deep image understanding completed successfully.",
            "image_hash": image_understanding["image"]["hash"],
            "image_analysis": image_understanding.get("image_analysis", {}),
            "text_analysis": neural_prediction.get("text_analysis", {}),
            "cross_modal_analysis": neural_prediction.get("cross_modal_analysis", {}),
            "final_prediction": neural_prediction.get("final_prediction", {}),
            "prediction": combined_prediction,
            "performance": {
                "cache_hit": image_understanding["performance"]["cache_hit"],
                "cache_type": image_understanding["performance"]["cache_type"],
                "processing_time_ms": image_understanding["performance"]["processing_time_ms"],
                "image_processing_time_ms": image_understanding["performance"].get("image_processing_time_ms", 0),
                "vision_inference_time_ms": image_understanding["performance"].get("vision_inference_time_ms", 0),
                "ocr_time_ms": image_understanding["performance"].get("ocr_time_ms", 0),
                "text_processing_time_ms": neural_prediction.get("metrics", {}).get("text_processing_time_ms", 0),
                "cross_modal_reasoning_time_ms": neural_prediction.get("metrics", {}).get("cross_modal_reasoning_time_ms", 0),
                "total_response_time_ms": image_understanding["performance"]["processing_time_ms"] + neural_prediction.get("metrics", {}).get("inference_time_ms", 0)
            }
        }

    except Exception as e:
        print(f"[Backend Error /analyze] {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Pipeline execution error: {str(e)}"}
        )


@app.post("/feedback")
async def record_feedback(
    image_hash: str = Form(...),
    user_sentiment: str = Form(...),
    predicted_sentiment: str = Form(...),
    match_score: int = Form(50),
    feedback_label: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Logs user corrections and evaluations for ongoing quality tracking."""
    try:
        cached = _memory_cache.get(image_hash, {})
        res = log_analysis_feedback(
            image_hash=image_hash,
            image_analysis=cached,
            user_sentiment=user_sentiment,
            predicted_sentiment=predicted_sentiment,
            match_score=match_score,
            user_feedback_label=feedback_label,
            notes=notes
        )
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@app.get("/feedback-metrics")
def get_feedback_metrics():
    """Returns evaluation metrics on recorded test cases."""
    return evaluate_feedback_metrics()


@app.post("/train-case")
async def train_case(
    text: str = Form(...),
    label: str = Form(...),
    image: UploadFile = File(...),
    epochs: int = Form(15)
):
    """Continuous Learning Endpoint."""
    try:
        from ml.case_trainer import train_on_new_case
        contents = await image.read()
        if len(contents) == 0:
            return JSONResponse(status_code=400, content={"success": False, "message": "Uploaded image file is empty."})

        result = train_on_new_case(text=text, image_bytes=contents, label=label, epochs=epochs)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Training error: {str(e)}"})
