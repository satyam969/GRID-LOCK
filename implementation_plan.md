# Implementation Plan: Deep ONNX Optimization & INT8 Quantization

You are absolutely right. While downsizing to `yolov8n` solved the immediate timeout ceiling, to achieve true production-ready 5-15s CPU inference, we must fully execute the optimizations detailed in your **TrafficGuard AI Optimization Guide**, specifically migrating the neural networks from PyTorch eager mode to **ONNX Runtime** with **INT8 Quantization**.

## Open Questions
- The optimization guide provides a custom `ONNXDetector` class for bounding boxes. However, because we recently added **Pose Estimation** (which outputs a complex 3D skeletal tensor rather than just 2D bounding boxes), writing a custom numpy post-processor for the pose model is highly error-prone. 
- **Recommendation:** We can still export the models to ONNX INT8, but load them using the `YOLO("model_int8.onnx", task="pose")` wrapper. Ultralytics natively uses `onnxruntime` under the hood when given an `.onnx` file, achieving the exact same C++ speedup while perfectly handling the complex pose keypoint math for us. Do you approve of this approach?

## Proposed Changes

### 1. Environment Updates
#### [MODIFY] [requirements.txt](file:///d:/trafficguard-ai/trafficguard-ai/backend/requirements.txt)
- Add `onnx` and `onnxruntime` dependencies required for graph compilation and optimized CPU execution.

### 2. Export & Quantization Pipeline
#### [NEW] [backend/scripts/optimize_models.py](file:///d:/trafficguard-ai/trafficguard-ai/backend/scripts/optimize_models.py)
- A new utility script that will:
  1. Export `yolov8n`, `yolov8n-pose`, `helmet_best`, `seatbelt_best`, and `plate_best` to ONNX format with dynamic batching.
  2. Use `onnxruntime.quantization.quantize_dynamic` to convert the FP32 weights into 8-bit integers (INT8), shrinking model size by 4x and boosting CPU throughput by an additional 2-3x.

### 3. Inference Engine Update
#### [MODIFY] [backend/app/core/detector.py](file:///d:/trafficguard-ai/trafficguard-ai/backend/app/core/detector.py)
- Refactor the `ModelRegistry` to point to the new `*_int8.onnx` optimized files instead of the native PyTorch `.pt` files.
- The rest of the pipeline (`violation_engine.py`) requires zero changes because the Ultralytics wrapper maintains the exact same prediction API.

## Verification Plan
1. Install the ONNX dependencies.
2. Run `optimize_models.py` and verify it successfully generates the `.onnx` and `_int8.onnx` files in the `models_weights` directory.
3. Start the Uvicorn server and process an image via the React frontend.
4. Verify in the backend logs that the `inference_time_ms` drops significantly (targeting under 10 seconds).

> [!IMPORTANT]
> Please review the plan above. If you approve, I will install the packages, run the quantization script, and wire up the ONNX runtime immediately!
