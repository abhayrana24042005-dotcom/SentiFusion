# SentiFusion: Multimodal Sentiment Analysis

**SentiFusion** is an end-to-end multimodal sentiment analysis research application. The system processes paired text and visual inputs through deep learning encoders and an attention-based fusion mechanism to predict sentiment categories (**Positive**, **Neutral**, **Negative**).

---

## Phase 1 Status: Multimodal Pipeline Foundation

In accordance with Phase 1 project requirements:
- **No fake sentiment predictions**: The application strictly validates multimodal inputs without mock/simulated classification.
- **Real end-to-end communication**: React frontend connects directly with the FastAPI backend over `multipart/form-data`.
- **Image validation**: Uploaded images are verified for format, size, and binary integrity using Pillow.
- **Instant preview**: Client-side preview with replace and remove capabilities.

---

## Directory Architecture

```
SentiFusion/
├── frontend/                     # React + Vite user interface
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx        # Research title, badge, and description
│   │   │   ├── TextInput.jsx     # Text input area, sample chips, char count
│   │   │   ├── ImageUpload.jsx   # Drag-and-drop & file picker
│   │   │   ├── ImagePreview.jsx  # Aspect-ratio preview, metadata, replace/remove
│   │   │   ├── AnalyzeButton.jsx # Prominent action button & spinner
│   │   │   └── AnalysisStatus.jsx# Success, loading, and error status panels
│   │   ├── styles/
│   │   │   └── App.css           # Premium dark AI research design system
│   │   ├── App.jsx               # Application state & API dispatch
│   │   └── main.jsx              # React root entry
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── backend/                      # Python FastAPI service
│   ├── main.py                   # /analyze endpoint, CORS, Pillow validation
│   ├── requirements.txt          # Python dependencies
│   └── README.md
│
├── ml/                           # Future Machine Learning architecture
│   └── README.md                 # Text Encoder + Image Encoder + Attention Fusion spec
│
├── data/                         # Future Dataset management
│   └── README.md                 # Paired sample schema, benchmarks (MVSA, Twitter)
│
├── docs/                         # Technical documentation
│   └── architecture_and_api.md   # Data flow, API contracts, verification checklist
│
└── README.md                     # Root project documentation
```

---

## Installation & Setup

### Prerequisites
- **Node.js** (v18 or higher) & **npm**
- **Python** (v3.10 or higher) & **pip**

### 1. Backend Setup
```bash
# Open terminal 1: Navigate to backend directory
cd backend

# (Recommended) Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt

# Start backend server
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Backend will be available at:
- **API Base**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`

---

### 2. Frontend Setup
```bash
# Open terminal 2: Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```
Frontend will be accessible at:
- **Local URL**: `http://127.0.0.1:5173`

---

## Verification & User Flow (Phase 1)

1. Open `http://127.0.0.1:5173` in your browser.
2. Enter text into the text area (or click the quick sample *"Absolutely loved this product!"*).
3. Upload an image (JPG, JPEG, PNG, or WEBP).
4. Notice the image preview immediately appears with filename, size, and Replace/Remove buttons.
5. Click **Analyze**.
6. Observe the loading state: *"Analyzing..."*.
7. The backend receives both modalities via `multipart/form-data`, validates them, and returns:
   ```json
   {
     "success": true,
     "message": "Text and image received successfully.",
     "text_received": true,
     "image_received": true
   }
   ```
8. The frontend renders the success state:
   - **Text and image received successfully.**
   - **✓ Text received**
   - **✓ Image received**
   - **Multimodal analysis pipeline is ready.**

---

## Testing Error Handling

- **Missing Text**: Select an image without entering text → click Analyze → validation prompt displays: *"Text input is required before analyzing."*
- **Missing Image**: Enter text without selecting an image → click Analyze → validation prompt displays: *"An image file is required before analyzing."*
- **Backend Offline**: Stop the FastAPI server and click Analyze → friendly network error notification is displayed.
- **Corrupted Image Guard**: Attempting to upload a corrupted or non-image binary returns an HTTP 400 error from the backend.
