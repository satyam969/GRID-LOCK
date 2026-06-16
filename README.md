# 🚦 TrafficGuard AI — Automated Traffic Violation Detection System

TrafficGuard AI is a computer vision-based traffic safety monitoring system designed for the Flipkart Grid-Lock Hackathon Phase 2. It automatically processes traffic images, detects vehicles and road users, identifies traffic violations, extracts license plate details using OCR, and aggregates analytics.

## 🏗️ System Architecture

```
                       +-----------------------------+
                       |  Traffic Camera / Uploader  |
                       +--------------+--------------+
                                      | (Image / Video)
                                      v
                       +--------------+--------------+
                       | Preprocessing Engine (CLAHE)|
                       +--------------+--------------+
                                      |
                                      v
                       +--------------+--------------+
                       | YOLOv8 Multi-Model Cascade  |
                       | - General (yolov8m)         |
                       | - Pose (yolov8m-pose)       |
                       | - Helmet / Seatbelt / Plate |
                       +--------------+--------------+
                                      | (Detections)
                                      v
                       +--------------+--------------+
                       |      Violation Engine       |
                       |  (Geometric Rule Checking)  |
                       +--------------+--------------+
                               /              \
                              v                v
                 +------------+-----------+  +-+----------+------------+
                 | License Plate EasyOCR  |  |    Evidence Annotator   |
                 +------------+-----------+  +------------+------------+
                              \                /
                               v              v
                       +--------------+--------------+
                       |     SQLAlchemy / SQLite     |
                       +--------------+--------------+
                                      |
                                      v
                       +--------------+--------------+
                       |  FastAPI REST / Websocket   |
                       +--------------+--------------+
                                      |
                                      v
                       +--------------+--------------+
                       |    React TS Dashboard UI    |
                       +-----------------------------+
```

### Cascade of Pre-trained Weights (No training from scratch):
1. **General Detector**: `yolov8m.pt` (Ultralytics official COCO weights) detects vehicles, pedestrians, and traffic lights.
2. **Pose Estimator**: `yolov8m-pose.pt` (Ultralytics official pose weights) extracts keypoints for triple-riding analysis.
3. **Helmet Detector**: Pre-trained custom weights (`helmet_best.pt`) downloaded programmatically.
4. **Seatbelt Detector**: Pre-trained custom weights (`seatbelt_best.pt`) downloaded programmatically.
5. **License Plate Detector**: Pre-trained custom weights (`plate_best.pt`) downloaded programmatically.

---

## ⚡ Quick Start (Docker Compose)

The easiest way to run the entire system (both backend and frontend) is using Docker Compose:

```bash
# Clone the repository and run compose
docker-compose up --build
```

- **Frontend UI**: [http://localhost](http://localhost) (mapped on port 80 via Nginx proxy)
- **FastAPI API Docs**: [http://localhost/api/docs](http://localhost/api/docs) (port 8000)

---

## 🔧 Local Manual Installation

### Backend Setup (FastAPI)
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the environment configuration:
   ```bash
   cp .env.example .env
   ```
4. Run the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup (Vite + React)
1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install Node packages:
   ```bash
   npm install
   ```
3. Run the frontend development server:
   ```bash
   npm run dev
   ```
4. Open your browser to [http://localhost:5173](http://localhost:5173).

---

## 📋 Run Automated Tests

The system comes with an incremental testing suite split into three phases:
- **Test 01**: Preprocessor CLAHE, model registry, and general vehicle bounding-box detection tests.
- **Test 02**: Unit testing individual violation detection logic (helmet, seatbelt, triple riding, wrong side, red-light, stop-line crossing) using mocked inputs.
- **Test 03**: REST API endpoint payloads, multipart image uploads, paginated database queries, and analytics filters.

Run the tests using pytest in the project root:
```bash
python -m pytest backend/tests/ -v
```
# GRID-LOCK
