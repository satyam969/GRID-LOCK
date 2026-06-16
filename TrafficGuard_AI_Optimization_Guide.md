# TrafficGuard AI — Deep Optimization & Production Architecture Guide

> **Version:** 2.0 Production-Ready  
> **Date:** June 2026  
> **Target:** Reduce inference from ~120-180s to 5-15s (CPU) / <1s (GPU)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture & Bottlenecks](#2-current-architecture--bottlenecks)
3. [Model-Level Optimizations](#3-model-level-optimizations)
4. [Cascaded Inference Pipeline](#4-cascaded-inference-pipeline)
5. [Batch Inference](#5-batch-inference)
6. [Parallel Model Execution](#6-parallel-model-execution)
7. [ROI-Based Processing Extensions](#7-roi-based-processing-extensions)
8. [Image Resolution Optimization](#8-image-resolution-optimization)
9. [OCR Optimization](#9-ocr-optimization)
10. [CLAHE & Preprocessing Pipeline](#10-clahe--preprocessing-pipeline)
11. [Model Preloading (Singleton Pattern)](#11-model-preloading-singleton-pattern)
12. [Async Architecture: Celery + Redis](#12-async-architecture-celery--redis)
13. [Frontend Polling (React)](#13-frontend-polling-react)
14. [Multi-Task Unified Model](#14-multi-task-unified-model)
15. [Feature Reuse / Shared Backbone](#15-feature-reuse--shared-backbone)
16. [Smart Early Exit Logic](#16-smart-early-exit-logic)
17. [Pipeline Scheduling & Priority Queue](#17-pipeline-scheduling--priority-queue)
18. [SQLite / Redis Job Store](#18-sqlite--redis-job-store)
19. [Deployment: Docker, Nginx & Environment](#19-deployment-docker-nginx--environment)
20. [Expected Performance Benchmarks](#20-expected-performance-benchmarks)
21. [Step-by-Step Implementation Roadmap](#21-step-by-step-implementation-roadmap)

---

## 1. Executive Summary

TrafficGuard AI currently runs **5 sequential PyTorch models** (General YOLO, Pose, Helmet, Seatbelt, License Plate + EasyOCR) on full 1080p images using CPU-only infrastructure. This results in **120-180 seconds per image**, triggering HTTP timeouts and `502 Bad Gateway` errors.

This guide provides **21 production-grade optimizations** — each with exact, copy-paste Python code — that will reduce inference to **5-15 seconds on CPU** and **<1 second on GPU**, while simultaneously improving reliability, scalability, and maintainability.

### Key Wins at a Glance

| Optimization | Expected Speedup | Effort |
|---|---|---|
| ONNX Runtime Export | 2-3x | Low |
| INT8 Quantization | 2-4x | Low |
| Model Downsizing (m to n) | 3-5x | Low |
| Cascaded Inference | 2-4x | Medium |
| Batch Inference | 1.3-2x | Medium |
| Parallel Execution | 1.5-2x | Medium |
| Resolution Downscale | 1.5-2x | Low |
| OCR Optimization | 2-3x | Medium |
| Celery + Redis Queue | Eliminates timeouts | High |
| Unified Multi-Task Model | 4-5x | High |

> **Combined realistic estimate:** 120s to 8-15s (CPU), to <1s (GPU)

---

## 2. Current Architecture & Bottlenecks

### 2.1 Current Pipeline (Sequential)

```
[1080p Image]
    |
    v
+----------------------------+
| yolov8m.pt (COCO)          |  <-- ~30-40s on CPU
| General Object Detector    |
| Cars, Motorcycles, Buses   |
| Persons, Trucks            |
+------------+---------------+
             |
             v
+----------------------------+
| yolov8n-pose.pt            |  <-- ~15-25s on CPU
| Pose Estimation            |
| 17 Skeletal Keypoints      |
+------------+---------------+
             |
             v
+----------------------------+
| helmet_best.pt             |  <-- ~10-20s on CPU
| Helmet Detection           |
| (on every person crop)     |
+------------+---------------+
             |
             v
+----------------------------+
| seatbelt_best.pt           |  <-- ~10-20s on CPU
| Seatbelt Detection         |
| (on every car crop)        |
+------------+---------------+
             |
             v
+----------------------------+
| EasyOCR                    |  <-- ~30-50s on CPU
| License Plate OCR          |
| CLAHE + Bilateral Filter   |
+----------------------------+
```

**Total Sequential Time: ~120-180 seconds**

### 2.2 Root Cause Analysis

| Bottleneck | Details |
|---|---|
| **Arithmetic Ceiling** | YOLOv8m = ~79 billion MACs per frame. CPU physically cannot process this fast. |
| **Memory Wall** | Model weights exceed L1/L2 cache. CPU stalls waiting for RAM (300+ cycle latency). |
| **Framework Overhead** | PyTorch eager mode = line-by-line execution, no graph optimization. |
| **Sequential Execution** | All 5 models run one after another with zero parallelism. |
| **Full Resolution** | 1080p passed to every model. Wasted compute on background pixels. |
| **No Conditional Logic** | Helmet model runs even when no motorcycle is found. |
| **EasyOCR Overhead** | Heavy LSTM-based recognition + CRAFT detection = slowest single component. |

---

## 3. Model-Level Optimizations

### 3.1 Export to ONNX Runtime

**Why:** ONNX Runtime uses static graph optimization, operator fusion, and platform-specific acceleration (Intel MKL-DNN on CPU). Benchmarks show 1.5-3x speedup over native PyTorch.

```python
# ============================================================
# export_models_to_onnx.py
# One-time script to export all YOLOv8 models to ONNX format
# ============================================================
from ultralytics import YOLO

MODELS_TO_EXPORT = {
    "general": "yolov8n.pt",       # Use yolov8n instead of yolov8m
    "pose": "yolov8n-pose.pt",
    "helmet": "helmet_best.pt",
    "seatbelt": "seatbelt_best.pt",
}

for name, model_path in MODELS_TO_EXPORT.items():
    print(f"[EXPORT] Exporting {name}: {model_path} -> ONNX")
    model = YOLO(model_path)

    # Export with dynamic axes for variable batch size
    model.export(
        format="onnx",
        imgsz=640,            # Standardize input size
        half=False,           # Keep FP32 for CPU (use True for GPU)
        simplify=True,        # Run onnx-simplifier to remove redundant ops
        dynamic=True,         # Allow variable batch sizes
        opset=17,             # Latest ONNX opset for best compatibility
    )
    print(f"[DONE] {name} exported successfully.\n")

print("All models exported to ONNX format.")
```

### 3.2 Run Inference with ONNX Runtime

```python
# ============================================================
# onnx_inference.py
# Production inference using ONNX Runtime instead of PyTorch
# ============================================================
import onnxruntime as ort
import numpy as np
import cv2


class ONNXDetector:
    """
    High-performance YOLO detector using ONNX Runtime.
    Replaces native PyTorch inference for 2-3x CPU speedup.
    """

    def __init__(self, onnx_path: str, input_size: int = 640):
        self.input_size = input_size

        # Configure ONNX session for maximum CPU performance
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.intra_op_num_threads = 4   # Match your CPU core count
        session_options.inter_op_num_threads = 2   # Cross-op parallelism
        session_options.enable_mem_pattern = True   # Optimize memory allocation
        session_options.enable_cpu_mem_arena = True  # Pre-allocate memory pools

        # Create session with CPU execution provider
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        # Cache input/output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize, normalize, transpose, and add batch dimension."""
        resized = cv2.resize(
            img, (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        blob = resized[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob

    def detect(self, img: np.ndarray, conf_threshold: float = 0.25) -> list:
        """Run detection on a single image."""
        input_tensor = self.preprocess(img)
        outputs = self.session.run(
            self.output_names, {self.input_name: input_tensor}
        )
        return self._postprocess(outputs, img.shape, conf_threshold)

    def detect_batch(self, images: list, conf_threshold: float = 0.25) -> list:
        """Run detection on a batch of images for higher throughput."""
        batch = np.concatenate(
            [self.preprocess(img) for img in images], axis=0
        )
        outputs = self.session.run(
            self.output_names, {self.input_name: batch}
        )
        return outputs

    def _postprocess(self, outputs, original_shape, conf_threshold):
        """Parse YOLO ONNX output into bounding box detections."""
        predictions = outputs[0]
        detections = []
        h_orig, w_orig = original_shape[:2]
        scale_x = w_orig / self.input_size
        scale_y = h_orig / self.input_size

        for pred in predictions[0]:
            conf = float(pred[4])
            if conf < conf_threshold:
                continue
            x1 = int(pred[0] * scale_x)
            y1 = int(pred[1] * scale_y)
            x2 = int(pred[2] * scale_x)
            y2 = int(pred[3] * scale_y)
            cls_id = int(pred[5])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
            })
        return detections


# Usage
if __name__ == "__main__":
    detector = ONNXDetector("yolov8n.onnx", input_size=640)
    img = cv2.imread("test_traffic.jpg")
    results = detector.detect(img)
    for det in results:
        print(f"Class: {det['class_id']}, Conf: {det['confidence']:.2f}, "
              f"Box: {det['bbox']}")
```

### 3.3 INT8 Quantization (Post-Training)

**Why:** INT8 reduces model size by 4x and speeds up CPU inference by 2-4x with minimal accuracy loss (~0.5% mAP drop typically).

```python
# ============================================================
# quantize_models.py
# Post-training INT8 quantization using ONNX Runtime
# ============================================================
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

ONNX_MODELS = {
    "general": "yolov8n.onnx",
    "helmet": "helmet_best.onnx",
    "seatbelt": "seatbelt_best.onnx",
}

for name, onnx_path in ONNX_MODELS.items():
    output_path = onnx_path.replace(".onnx", "_int8.onnx")
    print(f"[QUANTIZE] {name}: {onnx_path} -> {output_path}")

    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,  # 8-bit unsigned integer weights
        optimize_model=True,
    )

    orig_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[DONE] {name}: {orig_size:.1f}MB -> {quant_size:.1f}MB "
          f"({(1 - quant_size/orig_size)*100:.1f}% reduction)\n")
```

### 3.4 Model Downsizing: YOLOv8m to YOLOv8n

**Why:** YOLOv8m has ~25.9M parameters. YOLOv8n has ~3.2M parameters (8x fewer) with only ~3-5% mAP drop. On CPU, this alone can reduce inference from ~35s to ~5s.

```python
# ============================================================
# downsize_model.py
# Switch from yolov8m to yolov8n and re-export
# ============================================================
from ultralytics import YOLO

# BEFORE: Heavy model
# model = YOLO("yolov8m.pt")  # 25.9M params, ~49.0 GFLOPs

# AFTER: Lightweight model (8x smaller)
model = YOLO("yolov8n.pt")  # 3.2M params, ~8.7 GFLOPs

# Verify the model works on a test image
results = model.predict("test_traffic.jpg", conf=0.25)
print(f"Detections found: {len(results[0].boxes)}")

# Export the lightweight model to ONNX
model.export(format="onnx", imgsz=640, simplify=True, dynamic=True)
print("YOLOv8n exported to ONNX successfully.")
```

**Model Comparison Table:**

| Model | Params | GFLOPs | CPU Inference (1080p) |
|---|---|---|---|
| YOLOv8n | 3.2M | 8.7 | ~5s |
| YOLOv8s | 11.2M | 28.6 | ~12s |
| YOLOv8m | 25.9M | 78.9 | ~35s |
| YOLOv8l | 43.7M | 165.2 | ~60s |
| YOLOv8x | 68.2M | 257.8 | ~90s |

### 3.5 Export to OpenVINO (Best for Intel CPUs)

**Why:** OpenVINO is Intel's own inference toolkit. On Intel Xeon / Core CPUs, it outperforms ONNX Runtime by leveraging AVX-512, VNNI, and AMX instructions.

```python
# ============================================================
# export_openvino.py
# Export YOLOv8 to OpenVINO IR format for Intel CPUs
# ============================================================
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Export to OpenVINO — creates a folder with .xml and .bin files
model.export(
    format="openvino",
    imgsz=640,
    half=False,      # FP32 for CPU; set True for iGPU
    dynamic=True,
    int8=True,       # Enable INT8 quantization via NNCF
)
print("OpenVINO export complete: yolov8n_openvino_model/")

# Load and run inference with OpenVINO model
ov_model = YOLO("yolov8n_openvino_model/")
results = ov_model.predict("test_traffic.jpg")
print(f"OpenVINO detections: {len(results[0].boxes)}")
```

---
