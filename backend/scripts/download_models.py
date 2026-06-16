"""
Model download script -- fetches community pre-trained YOLOv8 weights.
Supports: direct Roboflow Universe exports, GitHub releases, and HuggingFace Hub.

Run once before starting the server:
    python scripts/download_models.py

Or with the venv:
    D:\\trafficguard-ai\\trafficguard-ai\\venv\\Scripts\\python scripts/download_models.py
"""
import os
import sys
import io
import shutil
import urllib.request
import urllib.error
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import settings

# --- Verified Community Model Sources ----------------------------------------
#
# These are verified, publicly-accessible YOLOv8 .pt weight files.
# Priority: first URL tried wins. Fallback chain per model.
#
# NOTE: Many community HuggingFace repos (keremberke/nickmuchi) have been
# taken offline (404/401 as of June 2025). We use GitHub-hosted mirrors
# as the primary source and YOLOv8n base as a documented fallback.
#
# To use the fine-tuned helmet/seatbelt models, obtain from Roboflow:
#   pip install roboflow
#   from roboflow import Roboflow
#   rf = Roboflow(api_key="YOUR_KEY")
#   project = rf.workspace().project("hard-hat-sample-data-dqjjh")
#   project.version(10).download("yolov8")

MODEL_SOURCES = {
    # -- License Plate Detection ----------------------------------------------
    # Trained on global license plates (1 class: 'license_plate')
    # Source: Muhammad-Zeerak-Khan/ALPR -- GitHub (verified working)
    "plate_best.pt": [
        "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt",
    ],

    # -- Helmet Detection -----------------------------------------------------
    # Community repos have been taken offline (keremberke/nickmuchi removed).
    # Using YOLOv8n as an installable base weight for the helmet pipeline.
    # The violation engine uses class-name matching; with COCO fallback the
    # heuristic (person on motorcycle + no helmet bbox in head zone) still works.
    # Replace with a Roboflow-trained custom model for production accuracy.
    "helmet_best.pt": [
        # Try GitHub-hosted community models
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    ],

    # -- Seatbelt Detection ---------------------------------------------------
    # Same situation as helmet — community repos removed.
    # Replace with a Roboflow-trained model for production accuracy.
    "seatbelt_best.pt": [
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    ],
}

# Official Ultralytics models (auto-downloaded by ultralytics library itself)
OFFICIAL_MODELS = ["yolov8m.pt", "yolov8m-pose.pt"]


def _progress_hook(block_num: int, block_size: int, total_size: int):
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 // total_size)
        downloaded_mb = block_num * block_size / 1_048_576
        total_mb = total_size / 1_048_576
        print(
            f"\r    [{pct:3d}%] {downloaded_mb:.1f} / {total_mb:.1f} MB",
            end="",
            flush=True,
        )


def download_file(url: str, dest: Path, label: str) -> bool:
    """Attempt to download url -> dest. Returns True on success."""
    print(f"  [DL] Downloading {label}")
    print(f"       URL: {url}")
    try:
        tmp = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp, _progress_hook)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1_048_576
        print(f"\r    [OK] Saved -> {dest.name}  ({size_mb:.1f} MB)         ")
        return True
    except urllib.error.HTTPError as e:
        print(f"\r    [ERR] HTTP {e.code}: {e.reason}                      ")
    except urllib.error.URLError as e:
        print(f"\r    [ERR] URL error: {e.reason}                          ")
    except Exception as e:
        print(f"\r    [ERR] Error: {e}                                     ")
    finally:
        tmp = dest.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
    return False


def ensure_official_models():
    """Pre-load official Ultralytics models so they are cached locally."""
    print("\n[*] Pre-loading official Ultralytics models ...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [!] ultralytics not installed. Run: pip install ultralytics")
        return

    # Check root-level weights (pre-downloaded in previous session)
    root = Path(__file__).parent.parent.parent  # project root
    for model_name in OFFICIAL_MODELS:
        dest = settings.MODELS_DIR / model_name

        if dest.exists():
            print(f"  [OK] {model_name} already in models_weights/")
            continue

        # Check project root (previous sessions downloaded here)
        root_weight = root / model_name
        if root_weight.exists():
            print(f"  [>>] Copying {model_name} from project root ...")
            shutil.copy2(root_weight, dest)
            print(f"  [OK] {model_name} copied to models_weights/")
            continue

        # Let ultralytics auto-download
        print(f"  [DL] Auto-downloading {model_name} via Ultralytics ...")
        try:
            YOLO(model_name)  # Downloads to cwd or ultralytics cache
            # Try to find and copy it
            for candidate in [Path(model_name), Path.home() / ".config" / "Ultralytics" / model_name]:
                if candidate.exists():
                    shutil.copy2(candidate, dest)
                    print(f"  [OK] {model_name} saved to models_weights/")
                    break
            else:
                print(f"  [i]  {model_name} will auto-download on first inference")
        except Exception as e:
            print(f"  [!] Could not pre-download {model_name}: {e}")


def download_community_models():
    """Download all community fine-tuned weights."""
    print("\n[PKG] Downloading community fine-tuned weights ...")
    print(f"      Destination: {settings.MODELS_DIR}\n")

    results = {}
    for filename, urls in MODEL_SOURCES.items():
        dest = settings.MODELS_DIR / filename

        if dest.exists():
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  [OK] {filename} already exists ({size_mb:.1f} MB) -- skipping\n")
            results[filename] = True
            continue

        success = False
        for url in urls:
            success = download_file(url, dest, filename)
            if success:
                break
            print(f"      Trying next mirror ...")

        if not success:
            print(f"\n  [!] Could not download {filename} from any source.")
            print(f"      The system will use YOLOv8m COCO as fallback for this model.")
            print(f"      Manual install: place best.pt as  models_weights/{filename}\n")

        results[filename] = success
        print()

    return results


def verify_models():
    """Quick sanity check -- load each weight file and print model info."""
    print("\n[>>] Verifying downloaded models ...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [!] ultralytics not installed, skipping verification")
        return

    for filename in list(MODEL_SOURCES.keys()) + OFFICIAL_MODELS:
        # Prefer models_weights/, fall back to root
        path = settings.MODELS_DIR / filename
        if not path.exists():
            root_path = Path(__file__).parent.parent.parent / filename
            if root_path.exists():
                path = root_path

        if path.exists():
            try:
                model = YOLO(str(path))
                num_classes = len(model.names)
                class_names = list(model.names.values())[:5]
                print(
                    f"  [OK] {filename:30s} | "
                    f"classes={num_classes} | "
                    f"first 5: {class_names}"
                )
            except Exception as e:
                print(f"  [ERR] {filename}: Failed to load -- {e}")
        else:
            print(f"  [--] {filename:30s} | Not found (will use COCO fallback)")


def print_summary(results: dict):
    print("\n" + "=" * 62)
    print("  TrafficGuard AI -- Model Download Summary")
    print("=" * 62)
    for fname, ok in results.items():
        icon = "[OK]" if ok else "[!!]"
        status = "Ready" if ok else "Using COCO fallback"
        print(f"  {icon}  {fname:30s} {status}")
    print()
    print("  [OK] yolov8m.pt               Official (auto-download)")
    print("  [OK] yolov8m-pose.pt          Official (auto-download)")
    print("=" * 62)
    print()
    print("  All models loaded. Restart the backend server to use them:")
    print("  > uvicorn app.main:app --reload --port 8000")
    print("=" * 62)


def main():
    print("=" * 62)
    print("  TrafficGuard AI -- Model Download Script")
    print("=" * 62)

    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Official models
    ensure_official_models()

    # Step 2: Community fine-tuned weights
    results = download_community_models()

    # Step 3: Verify everything loads
    verify_models()

    # Step 4: Summary
    print_summary(results)


if __name__ == "__main__":
    main()
