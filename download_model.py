"""手动下载 faster-whisper base 模型到本地目录"""
import os
import requests
import sys

MIRROR = "https://hf-mirror.com"
REPO = "Systran/faster-whisper-base"
DEST = os.path.join(os.path.dirname(__file__), "models", "base")

# faster-whisper 需要的文件列表
FILES = [
    "model.bin",
    "config.json",
    "tokenizer.json",
    "vocabulary.txt",
]

os.makedirs(DEST, exist_ok=True)

for fname in FILES:
    url = f"{MIRROR}/{REPO}/resolve/main/{fname}"
    dest_path = os.path.join(DEST, fname)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        print(f"  [SKIP] {fname} already exists ({size_mb:.1f} MB)", flush=True)
        continue

    print(f"  Downloading {fname}...", flush=True)
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 404:
            print(f"    [SKIP] {fname} not found in repo (not needed)", flush=True)
            continue
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        sys.stdout.write(f"\r    {fname}: {downloaded//1024//1024}MB/{total//1024//1024}MB ({pct}%)")
                        sys.stdout.flush()

        size_mb = os.path.getsize(dest_path) / 1024 / 1024
        print(f"\n    [OK] {fname} ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"\n    [FAIL] {fname}: {e}", flush=True)
        sys.exit(1)

print(f"\nAll files downloaded to {DEST}", flush=True)

# Verify model.bin exists
model_bin = os.path.join(DEST, "model.bin")
if os.path.exists(model_bin):
    size_mb = os.path.getsize(model_bin) / 1024 / 1024
    print(f"model.bin: {size_mb:.1f} MB", flush=True)

# Test loading from local path
print("\nTesting local model load...", flush=True)
from faster_whisper import WhisperModel
model = WhisperModel(DEST, device="cpu", compute_type="int8")
print("Model loaded successfully from local path!", flush=True)

import numpy as np
silence = np.zeros(16000, dtype=np.float32)
segments, info = model.transcribe(silence, beam_size=1)
text = " ".join([s.text.strip() for s in segments]).strip()
print(f"Test transcribe: '{text}' lang={info.language}", flush=True)
print("ALL GOOD!", flush=True)
