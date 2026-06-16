"""
Trigger Roboflow dataset export and wait for it to be ready, then download.
Roboflow generates export ZIPs asynchronously; this script polls until ready.
"""
import requests, zipfile, io, time, sys
from pathlib import Path

API_KEY   = 'zN3HCltTtKg0EgEe7P2O'
DATASETS  = Path(r'D:\trafficguard-ai\trafficguard-ai\datasets')
DATASETS.mkdir(exist_ok=True)

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

EXPORT_TARGETS = {
    'helmet':   ('roboflow-universe-projects', 'hard-hat-universe',       1,  'yolov8'),
    'seatbelt': ('roboflow-universe-projects', 'seatbelt-detection', 1,  'yolov8'),
}


def trigger_and_wait(workspace, project, version, fmt, max_wait=300):
    """Trigger export generation, poll until the GCS file exists."""
    api_url = f'https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}'

    prev_link = None
    for poll in range(max_wait // 10):
        r = session.get(api_url, params={'api_key': API_KEY}, timeout=30)
        if r.status_code != 200:
            print(f'  API error {r.status_code}')
            return None

        data = r.json()
        if 'error' in data:
            print(f'  API error: {data["error"]}')
            return None

        link = data.get('export', {}).get('link')
        prog = data.get('progress', 0)

        if not link:
            print(f'  No link yet, progress={prog}% — waiting...')
            time.sleep(10)
            continue

        # New link generated — wait a few seconds for GCS propagation
        if link != prev_link:
            print(f'  Export link obtained (progress={prog}%): {link[:60]}...')
            prev_link = link
            time.sleep(8)  # GCS propagation delay

        # Try to fetch the zip
        r2 = session.get(link, timeout=60, allow_redirects=True)
        if r2.status_code == 200 and len(r2.content) > 10_000:
            print(f'  Downloaded {len(r2.content)/1048576:.1f} MB')
            return r2.content
        elif r2.status_code == 404:
            print(f'  GCS 404 (not generated yet, poll={poll+1}) — retrying...')
            time.sleep(15)  # Wait longer for GCS
        else:
            print(f'  Unexpected HTTP {r2.status_code}, size={len(r2.content)}')
            time.sleep(10)

    print(f'  Timed out after {max_wait}s')
    return None


def download_and_extract(name, workspace, project, version, fmt):
    dest = DATASETS / name
    # Check if already extracted
    if list(dest.rglob('data.yaml')) or list(dest.rglob('*.yaml')):
        print(f'  [SKIP] {name} already extracted at {dest}')
        return True

    dest.mkdir(exist_ok=True)
    print(f'  Triggering export for {workspace}/{project} v{version}...')
    content = trigger_and_wait(workspace, project, version, fmt)

    if content is None:
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            z.extractall(dest)
        yamls = list(dest.rglob('*.yaml'))
        print(f'  [OK] Extracted! YAML files: {yamls}')
        return True
    except zipfile.BadZipFile as e:
        print(f'  [ERR] Bad zip: {e}')
        print(f'  Content start: {content[:200]}')
        return False


if __name__ == '__main__':
    print('=' * 55)
    print('  Roboflow Export + Download (with polling)')
    print('=' * 55)

    results = {}
    for name, (ws, proj, ver, fmt) in EXPORT_TARGETS.items():
        print(f'\n[{name.upper()}] {ws}/{proj} v{ver}')
        results[name] = download_and_extract(name, ws, proj, ver, fmt)

    print('\n' + '=' * 55)
    for name, ok in results.items():
        print(f'  {"[OK]" if ok else "[!!]"}  {name}: {"Ready" if ok else "FAILED"}')
    print('=' * 55)

    if all(results.values()):
        print('\nDatasets ready! Run:')
        print('  python backend/scripts/train_models.py')
    else:
        sys.exit(1)
