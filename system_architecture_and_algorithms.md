# TrafficGuard AI: Deep Technical Architecture & Algorithms

This document provides a highly technical, code-level breakdown of the algorithms used to evaluate complex traffic violations on single-frame images, detailing our multi-model ONNX cascade and the geometric heuristics that power the violation engine.

---

## 1. Multi-Model Pipeline Architecture

The engine (`violation_engine.py`) takes an OpenCV NumPy image array and passes it through the `Detector` class. We utilize the Ultralytics YOLOv8 framework, fully quantized to INT8 ONNX, running in a parallel `ThreadPoolExecutor` to maximize CPU throughput.

1. **`yolov8n_int8.onnx` (COCO)**: General object detector. Used to find spatial coordinates of:
   - `class_id: 2` (Car), `class_id: 3` (Motorcycle), `class_id: 5` (Bus), `class_id: 7` (Truck)
   - `class_id: 0` (Person)
2. **`yolov8n-pose_int8.onnx`**: Human Pose Estimation model. Used as a secondary verifier for highly occluded passengers.
3. **`helmet_best_int8.onnx`**: A specialized YOLOv8 model trained purely to detect heads, classifying them as `With Helmet` or `Without Helmet`.
4. **`seatbelt_best_int8.onnx`**: A specialized YOLOv8 model trained to detect unbuckled occupants inside vehicle windshields.
5. **`plate_best_int8.onnx`**: Custom license plate extraction model.

---

## 2. Advanced Heuristics & Filtering

### A. Triple Riding: Dual-Model Union & Center-Point Mapping
**Challenge:** Counting people on a motorcycle requires distinguishing pedestrians walking *behind* the motorcycle from riders actively sitting *on* it. Furthermore, passengers sitting behind the driver are often heavily occluded, causing the general object detection model to completely miss them, resulting in false negatives.

**Solution: Pose Union & Euclidean Distance Mapping**
1. We run both `yolov8n_int8.onnx` (General) and `yolov8n-pose_int8.onnx` (Pose) in parallel. Pose models are significantly better at detecting overlapping human limbs/heads.
2. We calculate the geometric center `(x, y)` of every detected person and every detected motorcycle.
3. **Euclidean Distance Matching:** For every person, we find the closest motorcycle center. If the person's center is within a threshold radius (`1.5x` the width/height of the motorcycle), they are officially assigned to that motorcycle. This bypasses the need for strict bounding-box intersection, which fails when passengers lean far back.
4. We run this algorithm on both the General persons list and the Pose persons list, taking the **maximum assigned rider count** between the two. If the validated rider count reaches 3 or more, a `TRIPLE_RIDING` violation is triggered.

### B. Illegal Parking: Expanded Proximity Heuristic
**Challenge:** Standard object detection cannot inherently "read" a No Parking sign. Furthermore, detecting if a car is "illegally parked" vs "driving normally" in a still image is extremely difficult.

**Solution: Edge-Thresholding and Occupancy Verification**
1. The engine defines the "shoulders" of the road using an expanded `edge_percentage` threshold of **35%**. Any vehicle within the outer 35% of the left or right frame is flagged as potentially parked.
2. We then verify **occupancy** by checking if there are any pedestrians (`class_id: 0`) standing within a `120px` radius of the vehicle. If the vehicle is on the deep shoulder of the road and has no driver standing nearby, it is flagged with an `ILLEGAL_PARKING` violation.

### C. Helmet Non-Compliance Logic
**Challenge:** Associating a bare head with a specific motorcycle, because the motorcycle bounding box only covers the chassis/wheels, not the rider sitting on top of it.

**Solution: Top 40% Anchor Cropping & HSV Heuristics**
Instead of projecting mathematical zones, we crop the top 40% of the *Rider's* bounding box. We then use an HSV skin-tone masking algorithm to count exposed skin pixels. If more than 10% of the head region consists of exposed skin-tones (bypassing the need for heavy secondary classifier models), it is flagged as a No Helmet violation.

### D. Seatbelt Non-Compliance Logic
**Challenge:** Standard models detect the exterior of a car, causing false positives from reflections on the doors or background objects.

**Solution: Windshield Cropping**
1. When a car is detected, we calculate its upper quadrant (the windshield).
2. We crop the image to isolate just the windshield area: `crop_img = img[int(y1):int(y1 + h * 0.45), int(x1):int(x2)]`.
3. The `seatbelt_best_int8.onnx` model is run exclusively inside this cropped zone, preventing it from hallucinating seatbelt violations on the car's exterior.

### E. OCR Text Extraction (License Plates)
**Challenge:** Low contrast images, glare, and shadows cause the OCR engine to hallucinate characters or fail entirely.

**Solution: CLAHE & Bilateral Preprocessing**
1. The detector finds a license plate bounding box.
2. The image is cropped and passed to the OCR preprocessing pipeline.
3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** is applied to normalize shadows and enhance text edges.
4. **Bilateral Filtering** is applied to smooth out pixel noise while keeping text edges razor-sharp.
5. If the EasyOCR engine returns a confidence score below 30%, the text is rejected to prevent hallucinations.

---

## 3. The CPU Timeout Bottleneck & The Full ONNX Optimization Solution

### The Problem: Arithmetic Ceiling & Synchronous Blocking
Processing a single 1080p image through 5 PyTorch models (General, Pose, Helmet, Seatbelt, Plate) plus EasyOCR on a purely CPU-bound environment initially took between **120 to 150 seconds**.

Modern network proxies (like Vite's Dev Server, Nginx) and browsers have hard-coded idle timeouts (usually 60-120 seconds). When the CPU exceeded this time limit, the proxy forcefully killed the HTTP connection (`ECONNRESET` / `502 Bad Gateway`), causing the frontend to throw an **"Analysis Failed"** error.

### The Solution: Parallel Execution + ONNX Runtime + INT8 Quantization
To completely solve this, we architected a 4-step optimization pipeline to dramatically accelerate CPU performance:

1. **Parallel Inference via ThreadPoolExecutor:** Instead of running the 5 models sequentially, independent models (Pose, Helmet, Seatbelt, Plate) are dispatched into a multi-threaded pool, executing concurrently across CPU cores.
2. **Model Downsizing:** Replaced the heavy `yolov8m` (25.9M parameters) with the ultra-lightweight `yolov8n` (3.2M parameters) for general detection and pose estimation.
3. **ONNX Export:** Exported all 5 PyTorch computational graphs (`.pt` files) into the highly optimized C++ ONNX format.
4. **Dynamic INT8 Quantization:** Using `onnxruntime`, we converted the 32-bit floating point weights into 8-bit integers.

**The Results (File Size & Speed):**
- `yolov8n_int8.onnx`: 12.4MB -> 3.4MB (-72.2%)
- `yolov8n-pose_int8.onnx`: 13.1MB -> 3.8MB (-70.9%)
- `seatbelt_best_int8.onnx`: 99.2MB -> 25.5MB (-74.3%)
- `helmet_best_int8.onnx`: 12.0MB -> 3.5MB (-70.9%)
- `plate_best_int8.onnx`: 11.9MB -> 3.4MB (-71.6%)

**Total CPU Inference Time dropped from ~150 seconds to under 10 seconds (a 15x speedup),** completely bypassing the proxy timeout constraints and enabling real-time edge processing.
