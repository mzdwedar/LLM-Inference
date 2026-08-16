import json
import urllib.request
import os
import sys
from pathlib import Path

# Credentials come from the repo-root .env (gitignored), never from this file.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass  # fall back to whatever is already exported in the environment

api_key = os.environ.get("NGC_API_KEY")
if not api_key:
    sys.exit(
        "NGC_API_KEY is not set. Copy .env.example to .env and fill it in, "
        "or run: set -a; source .env; set +a"
    )

url = "https://api.ngc.nvidia.com/v2/resources/nim/meta/llama-3.1-8b-instruct/manifest"

req = urllib.request.Request(url)
if api_key:
    req.add_header("Authorization", f"Bearer {api_key}")

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        profiles = data.get("profiles", {})
        print(f"Found {len(profiles)} model profiles:\n")
        for profile_id, meta in profiles.items():
            backend = meta.get("engine", "unknown")
            tp = meta.get("tensor_parallel_size", 1)
            precision = meta.get("precision", "unknown")
            print(f"ID: {profile_id} | Backend: {backend} | Precision: {precision} | TP: {tp}")
except Exception as e:
    print(f"Error fetching profiles: {e}")