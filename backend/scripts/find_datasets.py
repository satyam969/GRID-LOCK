"""
Search Roboflow Universe and download datasets for training.
"""
import requests
import sys
import shutil
from pathlib import Path
from roboflow import Roboflow

API_KEY = 'zN3HCltTtKg0EgEe7P2O'
MODELS_DIR = Path(r'D:\trafficguard-ai\trafficguard-ai\backend\models_weights')
DATASETS_DIR = Path(r'D:\trafficguard-ai\trafficguard-ai\datasets')
DATASETS_DIR.mkdir(exist_ok=True)

rf = Roboflow(api_key=API_KEY)

# ── Search Roboflow Universe ─────────────────────────────────────────────────
def search_rf(query, n=8):
    url = 'https://api.roboflow.com/search'
    params = {'api_key': API_KEY, 'q': query, 'type': 'dataset', 'page': 0}
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    results = data.get('results', [])
    print(f'\n[Search: "{query}"] — {len(results)} results')
    for res in results[:n]:
        rid = res.get('id', '?')
        imgs = res.get('images', '?')
        classes = res.get('classes', [])
        print(f'  {rid} | images={imgs} | classes={classes}')
    return results[:n]

# ── Download via SDK ──────────────────────────────────────────────────────────
def try_download(workspace, project, version, dest_name, fmt='yolov8'):
    dest = DATASETS_DIR / dest_name
    dest.mkdir(exist_ok=True)
    try:
        print(f'  Trying {workspace}/{project} v{version}...')
        p = rf.workspace(workspace).project(project)
        v = p.version(version)
        ds = v.download(fmt, location=str(dest), overwrite=True)
        print(f'  [OK] Downloaded to {ds.location}')
        return ds.location
    except Exception as e:
        print(f'  [ERR] {str(e)[:100]}')
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Search for helmet datasets
    helmet_results = search_rf('helmet detection motorcycle')
    seatbelt_results = search_rf('seatbelt detection car driver')

    # Attempt downloads from known good projects
    print('\n=== Downloading Helmet Dataset ===')
    helmet_path = (
        try_download('roboflow-universe-projects', 'hard-hat-universe', 1, 'helmet') or
        try_download('roboflow-universe-projects', 'safety-helmets-detection', 2, 'helmet') or
        try_download('roboflow-100', 'hard-hat-sample-data', 2, 'helmet')
    )

    print('\n=== Downloading Seatbelt Dataset ===')
    seatbelt_path = (
        try_download('roboflow-universe-projects', 'seatbelt-detection', 1, 'seatbelt') or
        try_download('roboflow-100', 'seatbelt-detection', 1, 'seatbelt')
    )

    print('\n=== Results ===')
    print(f'Helmet dataset:   {helmet_path or "NOT DOWNLOADED"}')
    print(f'Seatbelt dataset: {seatbelt_path or "NOT DOWNLOADED"}')
