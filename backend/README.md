# SentiFusion Backend - Phase 1

This is the FastAPI backend service for **SentiFusion**, a multimodal sentiment-analysis application.

## Overview

In **Phase 1**, this backend exposes the `/analyze` endpoint to receive multimodal data (`text` and `image`) using `multipart/form-data`. It strictly validates input presence and image integrity before returning a structured confirmation of receipt.

In accordance with Phase 1 design constraints, **no fake sentiment prediction** is returned.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`:
  - `fastapi`
  - `uvicorn[standard]`
  - `python-multipart`
  - `pillow`

## Installation

```bash
# Navigate to backend directory
cd backend

# (Optional) Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Server

Start the FastAPI application with Uvicorn:

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

## API Endpoints

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "service": "SentiFusion Backend",
  "phase": 1,
  "pipeline_ready": true
}
```

### 2. Analyze Multimodal Input
- **Endpoint**: `POST /analyze`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `text`: String (required, non-empty)
  - `image`: Binary file (required, format: JPG, JPEG, PNG, or WEBP)

#### Success Response (200 OK):
```json
{
  "success": true,
  "message": "Text and image received successfully.",
  "text_received": true,
  "image_received": true
}
```

#### Error Response (400 Bad Request):
```json
{
  "success": false,
  "message": "Text is required."
}
```
or
```json
{
  "success": false,
  "message": "Image is required."
}
```
or
```json
{
  "success": false,
  "message": "Invalid or corrupted image file."
}
```
