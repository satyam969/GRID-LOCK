"""
TrafficGuard AI — Fine-tune YOLOv8n for helmet and seatbelt detection.

Run from project root (with venv active):
    python backend/scripts/train_models.py

Training time: ~20-40 min on CPU, ~5-10 min on GPU.
Output: backend/models_weights/helmet_best.pt and seatbelt_best.pt
"""
import sys
import shutil
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent          # backend/
MODELS_DIR   = BASE_DIR / 'models_weights'
DATASETS_DIR = BASE_DIR.parent / 'datasets'          # trafficguard-ai/datasets/

TRAIN_CONFIG = {
    'helmet': {
        'data_dir':    DATASETS_DIR / 'helmet',
        'output_name': 'helmet_best.pt',
        'epochs':      30,
        'imgsz':       640,
        'base_model':  'yolov8n.pt',
        'desc':        'Helmet / no-helmet detection',
    },
    'seatbelt': {
        'data_dir':    DATASETS_DIR / 'seatbelt',
        'output_name': 'seatbelt_best.pt',
        'epochs':      30,
        'imgsz':       640,
        'base_model':  'yolov8n.pt',
        'desc':        'Seatbelt / no-seatbelt detection',
    },
}


def find_yaml(data_dir: Path) -> Path | None:
    """Find the data.yaml in a Roboflow downloaded dataset."""
    for yaml in data_dir.rglob('*.yaml'):
        if yaml.name not in ('roboflow.yaml',):
            return yaml
    return None


def train_model(name: str, cfg: dict) -> bool:
    """Fine-tune YOLOv8n on the given dataset. Returns True on success."""
    from ultralytics import YOLO

    data_dir = cfg['data_dir']
    if not data_dir.exists():
        print(f'  [SKIP] Dataset not found: {data_dir}')
        print(f'         Run scripts/find_datasets.py first.')
        return False

    yaml_path = find_yaml(data_dir)
    if yaml_path is None:
        print(f'  [ERR] No data.yaml found in {data_dir}')
        return False

    print(f'  Dataset YAML: {yaml_path}')
    dest = MODELS_DIR / cfg['output_name']

    model = YOLO(cfg['base_model'])

    print(f'  Training {name} model for {cfg["epochs"]} epochs...')
    results = model.train(
        data=str(yaml_path),
        epochs=cfg['epochs'],
        imgsz=cfg['imgsz'],
        batch=-1,          # Auto batch size
        device='0' if _has_gpu() else 'cpu',
        project=str(BASE_DIR / 'runs'),
        name=name,
        exist_ok=True,
        verbose=False,
        plots=True,
    )

    # Copy best weights to models_weights/
    best = Path(results.save_dir) / 'weights' / 'best.pt'
    if best.exists():
        shutil.copy2(best, dest)
        size_mb = dest.stat().st_size / 1_048_576
        print(f'  [OK] {cfg["output_name"]} saved ({size_mb:.1f} MB)')
        return True
    else:
        print(f'  [ERR] best.pt not found after training at {best}')
        return False


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    print('=' * 60)
    print('  TrafficGuard AI — Model Fine-Tuning Script')
    print('=' * 60)
    print(f'  GPU available: {_has_gpu()}')
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, cfg in TRAIN_CONFIG.items():
        print(f'\n[{name.upper()}] {cfg["desc"]}')
        print(f'  Output: {MODELS_DIR / cfg["output_name"]}')
        results[name] = train_model(name, cfg)

    print('\n' + '=' * 60)
    print('  Training Summary')
    print('=' * 60)
    for name, ok in results.items():
        icon = '[OK]' if ok else '[!!]'
        status = 'Trained & saved' if ok else 'Failed / skipped'
        print(f'  {icon}  {name:12s} {status}')

    print('\n  Restart backend to load new weights:')
    print('  > uvicorn app.main:app --reload --port 8000')
    print('=' * 60)


if __name__ == '__main__':
    main()
