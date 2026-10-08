# SentiFusion: System Architecture & API Specification (Phase 1)

This document describes the design, end-to-end data flows, API contracts, and verification procedures for **Phase 1** of **SentiFusion**.

---

## 1. System Architecture Overview

```
+-----------------------------------------------------------------------------------+
|                              REACT FRONTEND (Vite)                                |
|                                                                                   |
|   +--------------------+     +---------------------+     +--------------------+   |
|   |     TextInput      |     |  ImageUpload / Drop |     |    ImagePreview    |   |
|   |  - State binding   |     |  - File selector    |     |  - Instant object  |   |
|   |  - Char counter    |     |  - Type checking    |     |    URL preview     |   |
|   |  - Sample chips    |     |  - Size guardrail   |     |  - Replace/Remove  |   |
|   +---------+----------+     +----------+----------+     +--------------------+   |
|             |                           |                                         |
|             +-------------+-------------+                                         |
|                           |                                                       |
|                           v                                                       |
|                 +-------------------+                                             |
|                 |   AnalyzeButton   |                                             |
|                 +---------+---------+                                             |
|                           | (Validates inputs & builds FormData)                  |
|                           v                                                       |
|           fetch('http://127.0.0.1:8000/analyze', { method: 'POST', body: formData }) |
+---------------------------|-------------------------------------------------------+
                            |
                            | HTTP POST multipart/form-data
                            | (text + image binary)
                            v
+-----------------------------------------------------------------------------------+
|                              FASTAPI BACKEND                                      |
|                                                                                   |
|   +--------------------+     +---------------------+     +--------------------+   |
|   |  CORS Middleware   | --> | Input Extractor     | --> | Strict Validators  |   |
|   |  - Allow localhost |     | - text: Form(...)   |     | 1. Text non-empty  |   |
|   |    origins         |     | - image: UploadFile |     | 2. Image present   |   |
|   +--------------------+     +---------------------+     | 3. Mime/Ext valid  |   |
|                                                          | 4. Pillow verify() |   |
|                                                          +---------+----------+   |
|                                                                    |              |
|                               +------------------------------------+              |
|                               |                                                   |
|                               v                                                   |
|             +-----------------------------------+                                 |
|             | 200 OK / 400 Bad Request Response |                                 |
|             +-----------------+-----------------+                                 |
+-------------------------------|---------------------------------------------------+
                                |
                                | JSON confirmation / error
                                v
+-----------------------------------------------------------------------------------+
|                              REACT FRONTEND STATUS                                |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   | AnalysisStatus Component:                                                 |   |
|   |  - Loading: "Analyzing..." spinner & progress bar                         |   |
|   |  - Error: Clean alert with specific error message                         |   |
|   |  - Success:                                                               |   |
|   |      * "Text and image received successfully."                            |   |
|   |      * ✓ Text received                                                    |   |
|   |      * ✓ Image received                                                   |   |
|   |      * "Multimodal analysis pipeline is ready."                           |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------+
```

---

## 2. End-to-End Data Flow

### A. Text Flow (Frontend → Backend)
1. **User Input**: The user enters a string (e.g. `"Absolutely loved this product!"`) into `TextInput.jsx`.
2. **State Storage**: The value is bound to the React `text` state variable via controlled component binding.
3. **Pre-Flight Validation**: When `Analyze` is clicked, `App.jsx` verifies `text.trim().length > 0`.
4. **Serialization**: `FormData.append('text', text.trim())` packages the string.
5. **Network Dispatch**: Dispatched as part of the `multipart/form-data` payload via standard `fetch` API.
6. **Backend Ingestion**: FastAPI receives `text: Optional[str] = Form(None)` in `backend/main.py`.
7. **Backend Validation**:
   - Ensures `text is not None`.
   - Ensures `len(text.strip()) > 0`.
   - Returns HTTP 400 `{ "success": false, "message": "Text is required." }` if invalid.

### B. Image Flow (Frontend → Backend)
1. **Selection / Drop**: The user selects or drags an image file into `ImageUpload.jsx`.
2. **Instant Preview**: A browser blob URL is created with `URL.createObjectURL(file)` and rendered by `ImagePreview.jsx`.
3. **Pre-Flight Validation**: `App.jsx` confirms `imageFile !== null` and format conforms to JPG, JPEG, PNG, or WEBP.
4. **Serialization**: `FormData.append('image', imageFile)` adds the binary file object.
5. **Network Dispatch**: Dispatched with multipart boundary markers to the backend.
6. **Backend Ingestion**: FastAPI parses the multi-part stream into `image: Optional[UploadFile] = File(None)`.
7. **Backend Validation**:
   - Ensures `image.filename` is present.
   - Checks MIME type and extension against whitelist.
   - Reads image buffer bytes: `contents = await image.read()`.
   - Instantiates `io.BytesIO(contents)` and calls `Pillow`'s `Image.open(...).verify()` to guarantee valid, non-corrupted raster data.
   - Returns HTTP 400 with a descriptive error if corrupted or unsupported.

---

## 3. API Contract Reference

### `GET /health`
Returns service status and Phase 1 pipeline readiness.

#### Response:
```json
{
  "status": "healthy",
  "service": "SentiFusion Backend",
  "phase": 1,
  "pipeline_ready": true
}
```

---

### `POST /analyze`
Receives both modalities and verifies input integrity.

#### Request Headers:
```http
Content-Type: multipart/form-data
```

#### Request Fields:
| Field | Type | Description | Required |
|---|---|---|---|
| `text` | String | User text or review to analyze | Yes |
| `image` | Binary File | Uploaded JPG, JPEG, PNG, or WEBP image | Yes |

#### Successful Response (HTTP 200 OK):
```json
{
  "success": true,
  "message": "Text and image received successfully.",
  "text_received": true,
  "image_received": true
}
```

#### Error Responses (HTTP 400 Bad Request):
- **Missing Text**:
  ```json
  {
    "success": false,
    "message": "Text is required."
  }
  ```
- **Missing Image**:
  ```json
  {
    "success": false,
    "message": "Image is required."
  }
  ```
- **Unsupported Format**:
  ```json
  {
    "success": false,
    "message": "Unsupported image format. Allowed formats: JPG, JPEG, PNG, WEBP."
  }
  ```
- **Corrupted / Invalid File**:
  ```json
  {
    "success": false,
    "message": "Invalid or corrupted image file."
  }
  ```

---

## 4. Future ML Integration Point

In **Phase 2**, the verified input data will be handed off to the machine learning inference module before generating the response.

Location in `backend/main.py`:
```python
# [PHASE 1 COMPLETE]: Inputs validated
# [PHASE 2 HOOK]:
# from ml.inference import predict_multimodal_sentiment
# result = predict_multimodal_sentiment(text=text, image_bytes=contents)
# return {
#     "success": True,
#     "message": "Analysis completed successfully.",
#     "prediction": result["label"],       # "positive" | "neutral" | "negative"
#     "confidence": result["confidence"],  # e.g., 0.94
#     "text_received": True,
#     "image_received": True
# }
```

---

## 5. Phase 1 Testing Checklist

| Test Item | Action | Expected Outcome |
|---|---|---|
| **1. Empty Text Validation** | Leave text blank, select image, click Analyze | Frontend displays warning: *"Text input is required before analyzing."* Backend is not called. |
| **2. Missing Image Validation** | Enter text, leave image unselected, click Analyze | Frontend displays warning: *"An image file is required before analyzing."* |
| **3. Instant Image Preview** | Select a valid image file | Preview appears immediately with correct aspect ratio, file size, name, and Replace/Remove buttons. |
| **4. Image Replace & Remove** | Click Replace or Remove on preview | Remove clears image; Replace re-opens file dialog. |
| **5. Valid Multimodal Submission** | Enter text and select valid image, click Analyze | Loading indicator appears ("Analyzing..."), followed by HTTP 200 success card: *"Text and image received successfully."*, *"✓ Text received"*, *"✓ Image received"*. |
| **6. Backend Corrupted Image Guard** | Send arbitrary text file renamed as `.png` via curl | Backend returns HTTP 400 with *"Invalid or corrupted image file."* |
| **7. CORS Verification** | Frontend on port 5173 calls backend on port 8000 | Request completes with proper `Access-Control-Allow-Origin` headers without browser CORS block. |
| **8. Absence of Fake Predictions** | Inspect UI and API responses | No mock/random Positive/Neutral/Negative ratings or fake accuracies exist anywhere. |
