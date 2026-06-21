# 🚦 TrafficGuard AI — Automated Multi-Violation Detection & Intelligent Traffic Enforcement System

> **Flipkart GRID-LOCK Hackathon | Prototype Round 2**
> Theme: *Automated Photo Identification and Classification for Traffic Violations Using Computer Vision*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://ultralytics.com/)
[![ONNX](https://img.shields.io/badge/ONNX-INT8_Quantized-orange.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 💡 What is TrafficGuard AI?

**TrafficGuard AI** is a production-grade, end-to-end intelligent traffic enforcement platform that uses a **cascaded, multi-model AI pipeline** to detect 7 distinct traffic violations from a single camera frame — without any manual intervention.

Unlike systems relying on a single slow monolithic model, TrafficGuard AI runs **5 specialized neural networks in sequence**, each optimized for a specific task. The entire pipeline is quantized to **INT8 ONNX** format, slashing CPU inference time from **~150 seconds to under 10 seconds** (a **15x speedup**).

---

## 🚀 Key Features

### 🤖 AI Violation Detection (7 Violation Types)
| Violation | Detection Method |
|-----------|-----------------|
| **Triple Riding** | YOLOv8n-Pose skeletal hip-keypoint mapping to confirm riders seated on motorcycle |
| **Helmet Non-Compliance** | Custom INT8 helmet classifier + skin-tone HSV fallback heuristic |
| **Seatbelt Non-Compliance** | Windshield crop (top 45% of car bbox) fed to INT8 seatbelt classifier |
| **Red-Light Crossing** | HSV color masking on traffic light ROI + stop-line geometric threshold |
| **Stop-Line Encroachment** | Bounding box intersection ratio against configurable stop-line Y coordinate |
| **Wrong-Side Driving** | Frame bisection with traffic flow direction configuration (Keep Left / Keep Right) |
| **Illegal Parking** | Road-edge proximity heuristic + admin-configured No-Parking Zone polygons |

### 🏛️ Full Enforcement Workflow
- **Officer Moderation Panel** — Review, approve, or dismiss flagged incidents from a slide-out drawer
- **Automated Challan Issuance** — Generate e-tickets with fines mapped to the **Indian Motor Vehicle Act 2019** (Sec 194D, 194B, etc.)
- **Repeat Offender Tracking** — Relational database tracks violation count per plate; **doubles the fine** at 3+ violations
- **RTO Vehicle Lookup** — Autocomplete search by partial plate number to pull complete violation history

### 📊 System Administration
- **Camera Registry** — Register IoT cameras with GPS coordinates, zone assignments, and status tracking
- **AI Settings Panel** — Live sliders to tune confidence thresholds per model without restarting
- **Analytics Dashboard** — 7-day rolling charts, violation distribution pie charts, KPI cards
- **CSV Report Export** — Download daily incident summaries and challan revenue reports

---

## ⚡ Hardware Optimization: ONNX INT8 Quantization

Running 5 PyTorch models on a CPU originally took **~150 seconds per image**, breaking the browser's 120-second HTTP timeout. We solved this with a 3-step optimization:

| Step | Technique | Result |
|------|-----------|--------|
| 1 | **Model Downsizing** — YOLOv8m (25.9M params) → YOLOv8n (3.2M params) | ~8x param reduction |
| 2 | **ONNX Export** — PyTorch `.pt` → C++ ONNX computational graph | Faster C++ runtime |
| 3 | **INT8 Quantization** — FP32 weights → 8-bit integer weights | ~72% file size reduction |

**Results:**

| Model | Before | After | Reduction |
|-------|--------|-------|-----------|
| `yolov8n_int8.onnx` | 12.4 MB | 3.4 MB | -72.2% |
| `yolov8n-pose_int8.onnx` | 13.1 MB | 3.8 MB | -70.9% |
| `helmet_best_int8.onnx` | 12.0 MB | 3.5 MB | -70.9% |
| `seatbelt_best_int8.onnx` | 99.2 MB | 25.5 MB | -74.3% |
| `plate_best_int8.onnx` | 11.9 MB | 3.4 MB | -71.6% |

**Total inference time: ~150s → <10s (15x speedup)**

---

## 🏗️ System Architecture

```
  Traffic Camera / Image Upload
           |
           v
  ┌────────────────────────┐
  │  CLAHE Preprocessing   │  (Contrast enhancement for low-light images)
  └────────────┬───────────┘
               |
               v
  ┌────────────────────────────────────────────┐
  │         YOLOv8 Multi-Model Cascade         │
  │  ┌────────────┐   ┌────────────────────┐  │
  │  │ General    │   │ Pose Estimation    │  │
  │  │ yolov8n    │   │ yolov8n-pose       │  │
  │  │ (vehicles, │   │ (17 keypoints for  │  │
  │  │  persons,  │   │  triple-riding     │  │
  │  │  lights)   │   │  hip validation)   │  │
  │  └────────────┘   └────────────────────┘  │
  │  ┌──────────┐ ┌──────────┐ ┌───────────┐  │
  │  │ Helmet   │ │Seatbelt  │ │  License  │  │
  │  │ INT8     │ │  INT8    │ │  Plate    │  │
  │  │Classifier│ │Classifier│ │  INT8     │  │
  │  └──────────┘ └──────────┘ └───────────┘  │
  └────────────────────┬───────────────────────┘
                       |
                       v
  ┌────────────────────────────────────────────┐
  │           Violation Engine                 │
  │  (Geometric Rules + Heuristics)            │
  └─────────────────┬──────────────────────────┘
                    |
         ┌──────────┴──────────┐
         v                     v
  ┌──────────────┐    ┌──────────────────┐
  │  EasyOCR     │    │ Evidence Annotator│
  │  (CLAHE +    │    │ (Bounding boxes  │
  │  Bilateral   │    │  on annotated    │
  │  Filtering)  │    │  image)          │
  └──────┬───────┘    └───────┬──────────┘
         └──────────┬─────────┘
                    v
  ┌────────────────────────────────────────────┐
  │    SQLAlchemy Async / SQLite               │
  │  (Violations, Vehicles, Cameras,           │
  │   Challans, Officers, AuditLogs)           │
  └────────────────────┬───────────────────────┘
                       |
                       v
  ┌────────────────────────────────────────────┐
  │          FastAPI REST API                  │
  │       http://localhost:8000/docs           │
  └────────────────────┬───────────────────────┘
                       |
                       v
  ┌────────────────────────────────────────────┐
  │        React 18 + TypeScript + Vite        │
  │       http://localhost:5173                │
  └────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, Vite, TypeScript, Tailwind CSS, Recharts, Lucide Icons |
| **Backend** | FastAPI (Python 3.12), Uvicorn, SQLAlchemy (Async), SQLite |
| **AI/CV** | Ultralytics YOLOv8, EasyOCR, OpenCV (CLAHE + Bilateral Filter), NumPy |
| **Optimization** | ONNX Runtime, Dynamic INT8 Quantization |
| **Infrastructure** | Docker, Nginx (production), Vite DevServer (development) |

---

## 📁 Project Structure

```
GRID-LOCK/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI route handlers
│   │   │   ├── analytics.py  # KPI and trends endpoints
│   │   │   ├── cameras.py    # IoT camera registry
│   │   │   ├── challans.py   # Challan issuance & revenue
│   │   │   ├── upload.py     # Image analysis pipeline
│   │   │   ├── vehicles.py   # RTO lookup & repeat offenders
│   │   │   └── violations.py # Violation CRUD + moderation
│   │   ├── core/
│   │   │   ├── annotator.py          # Bounding box drawing
│   │   │   ├── detector.py           # ONNX model registry
│   │   │   ├── illegal_parking_detection.py
│   │   │   ├── ocr_engine.py         # EasyOCR + preprocessing
│   │   │   ├── preprocessor.py       # CLAHE + denoise
│   │   │   ├── red_light_detection.py
│   │   │   ├── violation_engine.py   # Main logic engine
│   │   │   └── wrong_side_detection.py
│   │   ├── models/           # SQLAlchemy ORM models
│   │   ├── schemas/          # Pydantic validation schemas
│   │   ├── config.py
│   │   ├── database.py
│   │   └── main.py
│   ├── models_weights/       # YOLO .pt and .onnx weight files
│   ├── scripts/
│   │   ├── download_models.py   # Auto-downloads community weights
│   │   └── optimize_models.py   # Exports & quantizes to INT8 ONNX
│   ├── tests/                   # 20-test pytest suite
│   ├── requirements.txt
│   └── settings.json            # Live AI threshold configuration
├── frontend/
│   ├── src/
│   │   ├── api/              # Axios API client
│   │   └── pages/
│   │       ├── Analyze.tsx         # Image upload & results
│   │       ├── Analytics.tsx       # Charts & KPI dashboard
│   │       ├── CameraManagement.tsx
│   │       ├── ChallanManagement.tsx
│   │       ├── Dashboard.tsx
│   │       ├── Reports.tsx
│   │       ├── Settings.tsx
│   │       ├── VehicleLookup.tsx
│   │       └── Violations.tsx
│   └── vite.config.ts
├── docker-compose.yml
├── start_servers.bat         # Windows one-click startup script
├── violation_detection_algorithms.md
├── walkthrough.md
└── system_architecture_and_algorithms.md
```

---

## ⚙️ Quick Start

### Prerequisites
- **Python 3.10 – 3.12** (added to PATH)
- **Node.js v18+** and npm
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/satyam969/GRID-LOCK.git
cd GRID-LOCK
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Download AI Model Weights
```bash
python backend/scripts/download_models.py
```
> This downloads `plate_best.pt` from verified GitHub mirrors, and the official `yolov8n.pt` / `yolov8n-pose.pt` from Ultralytics for the helmet and seatbelt pipelines.

### 4. Run the Backend
```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
📖 **Swagger API Docs:** http://127.0.0.1:8000/docs

### 5. Run the Frontend *(open a new terminal)*
```bash
cd frontend
npm install
npm run dev
```
🌐 **Web Dashboard:** http://localhost:5173

### 🪟 Windows Shortcut
Double-click **`start_servers.bat`** in the project root to launch both servers automatically in separate terminal windows.

---

## 🐳 Docker Compose (Production)
```bash
docker-compose up --build
```
- **UI:** http://localhost (Nginx on port 80)
- **API Docs:** http://localhost/api/docs

---

## 🧪 Running Tests
```bash
# From project root (with venv activated)
python -m pytest backend/tests/ -v
```
The test suite contains **20 tests** covering:
- CLAHE preprocessor and model registry
- Violation detection logic (helmet, seatbelt, triple riding, red-light, stop-line)
- REST API endpoints and paginated database queries

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [walkthrough.md](walkthrough.md) | Full technical walkthrough of all pages and the AI pipeline |
| [system_architecture_and_algorithms.md](system_architecture_and_algorithms.md) | Deep-dive into ONNX optimization and CPU timeout solutions |
| [violation_detection_algorithms.md](violation_detection_algorithms.md) | Precise heuristic and algorithm breakdown for all 7 violations |

---

## 🗺️ Roadmap / Future Scope
- [ ] **Real-time RTMP video stream** parsing (currently single-frame images)
- [ ] **Mobile officer app** (React Native) for field ticketing
- [ ] **National VAHAN/Parivahan API integration** for live RTO database lookups
- [ ] **GPU-accelerated inference** via CUDA ONNX Runtime provider
- [ ] **Multi-camera dashboard** with live violation heatmaps

---

## 👨‍💻 Team
Built for the **Flipkart GRID-LOCK Hackathon 2025** — Prototype Round 2.

---

## 📄 License
This project is licensed under the MIT License.
