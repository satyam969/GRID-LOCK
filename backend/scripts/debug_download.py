import requests, zipfile, io, time
from pathlib import Path

API_KEY   = 'zN3HCltTtKg0EgEe7P2O'
MODELS    = Path(r'D:\trafficguard-ai\trafficguard-ai\backend\models_weights')
DATASETS  = Path(r'D:\trafficguard-ai\trafficguard-ai\datasets')

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; TrafficGuardAI/1.0)'})

# ── Step 1: Get export link ───────────────────────────────────────────────────
url = 'https://api.roboflow.com/roboflow-universe-projects/hard-hat-universe/1/yolov8'
for attempt in range(5):
    r = session.get(url, params={'api_key': API_KEY}, timeout=30)
    data = r.json()
    link = data.get('export', {}).get('link')
    prog = data.get('progress', 0)
    if link:
        print(f'Export link: {link}')
        break
    print(f'Export generating {prog}%... (attempt {attempt+1})')
    time.sleep(8)
else:
    print('No link obtained')
    exit(1)

# ── Step 2: Download and inspect ──────────────────────────────────────────────
r2 = session.get(link, timeout=180, allow_redirects=True)
print(f'Status: {r2.status_code}')
ct = r2.headers.get('Content-Type', '')
cl = r2.headers.get('Content-Length', '?')
print(f'Content-Type: {ct}')
print(f'Content-Length: {cl}')
print(f'Body size: {len(r2.content)} bytes')
print(f'Body start: {r2.content[:300]}')

# Try to open as zip
if len(r2.content) > 1000:
    try:
        with zipfile.ZipFile(io.BytesIO(r2.content)) as z:
            names = z.namelist()[:10]
            print(f'ZIP OK - {len(z.namelist())} files, first 10: {names}')
            dest = DATASETS / 'helmet'
            dest.mkdir(parents=True, exist_ok=True)
            z.extractall(dest)
            print(f'Extracted to {dest}')
    except Exception as e:
        print(f'Not a valid zip: {e}')
        # Save for inspection
        Path('debug_response.bin').write_bytes(r2.content[:1000])
