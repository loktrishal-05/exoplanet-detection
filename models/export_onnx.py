"""Export the trained XGBoost booster to ONNX + a small JSON for browser inference.

    python models/export_onnx.py          # needs: xgboost onnxmltools onnxruntime

Writes models/saved/xgb_planet.onnx and web/model.json (medians, isotonic map,
threshold, presets with expected outputs), then verifies ONNX == XGBoost.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parent.parent
SAVED = ROOT / "models" / "saved"
report = json.loads((SAVED / "eval_report.json").read_text())
feats, med = report["features"], np.array(report["medians"])
cal = json.loads((SAVED / "xgb_calibration.json").read_text())
thr = json.loads((SAVED / "xgb_threshold.json").read_text())["threshold"]

booster = xgb.Booster()
booster.load_model(SAVED / "xgb_planet.json")
booster.feature_names = None  # onnxmltools wants f0..fN
onx = convert_xgboost(booster, initial_types=[("features", FloatTensorType([None, len(feats)]))],
                      target_opset=15)
(SAVED / "xgb_planet.onnx").write_bytes(onx.SerializeToString())
sess = ort.InferenceSession(str(SAVED / "xgb_planet.onnx"))


def onnx_raw(x):
    probs = sess.run(["probabilities"], {"features": np.asarray(x, np.float32)})[0]
    return np.asarray(probs, float)[:, 1]  # column 1 = P(planet)


def calibrated(raw):
    return float(np.clip(np.interp(raw, cal["x"], cal["y"]), 0, 1))


# presets: illustrative SYNTHETIC inputs = the model's most planet-like / most
# binary-like rows from a seeded random sweep around the training medians
rng = np.random.default_rng(0)
sweep = (med * rng.lognormal(0, 0.8, size=(20000, len(feats)))).astype(np.float32)
sc = onnx_raw(sweep)
presets = {"Planet-like (synthetic)": dict(zip(feats, sweep[sc.argmax()].tolist())),
           "Eclipsing-binary-like (synthetic)": dict(zip(feats, sweep[sc.argmin()].tolist()))}

X = med * rng.lognormal(0, 0.6, size=(64, len(feats)))
X = np.vstack([X] + [[[p[f] for f in feats]] for p in presets.values()]).astype(np.float32)
ref = booster.predict(xgb.DMatrix(X))
got = onnx_raw(X)
diff = float(np.max(np.abs(ref - got)))
print(f"ONNX vs XGBoost on {len(X)} rows: max abs diff = {diff:.2e}")
assert diff < 1e-5, "ONNX output does not match XGBoost"

out = {
    "features": feats, "medians": med.tolist(), "calibration": cal, "threshold": thr,
    "presets": {n: {"values": [p[f] for f in feats],
                    "raw": float(onnx_raw([[p[f] for f in feats]])[0]),
                    "prob": calibrated(float(onnx_raw([[p[f] for f in feats]])[0]))}
                for n, p in presets.items()},
}
(ROOT / "web" / "model.json").write_text(json.dumps(out, indent=1))
for n, p in out["presets"].items():
    print(n, "raw=%.4f calibrated=%.4f" % (p["raw"], p["prob"]))
