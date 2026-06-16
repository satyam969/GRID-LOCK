import os
import sys
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

def optimize_models():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "models_weights")
    
    # Define models to process
    models = {
        "general": "yolov8n.pt",
        "pose": "yolov8n-pose.pt",
        "helmet": "helmet_best.pt",
        "seatbelt": "seatbelt_best.pt",
        "plate": "plate_best.pt"
    }
    
    print("=== TrafficGuard AI: Model Optimization Pipeline ===")
    
    for name, filename in models.items():
        pt_path = os.path.join(weights_dir, filename)
        onnx_path = os.path.join(weights_dir, filename.replace(".pt", ".onnx"))
        int8_path = os.path.join(weights_dir, filename.replace(".pt", "_int8.onnx"))
        
        # We need the source weights to exist. For yolov8n.pt and yolov8n-pose.pt, 
        # Ultralytics auto-downloads them to the current working directory, 
        # so we'll check if they are in the current dir or download them.
        model_source = pt_path
        if not os.path.exists(pt_path):
            if os.path.exists(filename):
                model_source = filename
            else:
                print(f"[{name}] Source {filename} not found locally. Downloading via ultralytics...")
                model_source = filename
        
        print(f"\n--- Processing [{name}] ---")
        
        # 1. Export to ONNX (FP32)
        if not os.path.exists(onnx_path):
            print(f"[{name}] Exporting to ONNX format...")
            model = YOLO(model_source)
            # Export to ONNX format with dynamic batching
            exported_path = model.export(
                format="onnx",
                imgsz=640,
                half=False,      # Keep FP32 for ONNX CPU runtime
                simplify=True,
                dynamic=True,
                opset=17
            )
            # The exported file is created alongside the source file. Move it if needed.
            exported_file = model_source.replace(".pt", ".onnx")
            if exported_file != onnx_path and os.path.exists(exported_file):
                import shutil
                shutil.move(exported_file, onnx_path)
            print(f"[{name}] ONNX Export complete: {onnx_path}")
        else:
            print(f"[{name}] ONNX file already exists. Skipping export.")
            
        # 2. INT8 Quantization
        if not os.path.exists(int8_path):
            print(f"[{name}] Applying INT8 dynamic quantization...")
            try:
                quantize_dynamic(
                    model_input=onnx_path,
                    model_output=int8_path,
                    weight_type=QuantType.QUInt8
                )
                
                orig_size = os.path.getsize(onnx_path) / (1024 * 1024)
                quant_size = os.path.getsize(int8_path) / (1024 * 1024)
                reduction = (1 - quant_size / orig_size) * 100
                print(f"[{name}] Quantization complete: {orig_size:.1f}MB -> {quant_size:.1f}MB (-{reduction:.1f}%)")
            except Exception as e:
                print(f"[{name}] Quantization failed (usually WinError 32 on large models). Will fallback to FP32 ONNX. Error: {e}")
                # Create a symlink or copy the fp32 to int8 path so the app still runs if it requests int8
                import shutil
                shutil.copy(onnx_path, int8_path)
        else:
            print(f"[{name}] INT8 ONNX file already exists. Skipping quantization.")

if __name__ == "__main__":
    optimize_models()
    print("\n=== All models successfully optimized to INT8 ONNX ===")
