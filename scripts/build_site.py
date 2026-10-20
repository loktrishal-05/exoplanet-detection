"""Assemble the static site: python scripts/build_site.py  ->  dist/ (deploy this folder)."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)  # files below are overwritten in place
shutil.copy(ROOT / "preview" / "index.html", DIST / "index.html")
shutil.copy(ROOT / "models" / "saved" / "xgb_planet.onnx", DIST / "xgb_planet.onnx")
shutil.copy(ROOT / "web" / "model.json", DIST / "model.json")
print("built", sorted(p.name for p in DIST.iterdir()))
