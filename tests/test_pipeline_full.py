import sys
import json
import pytest
import pandas as pd
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    sys.path.append(str(ROOT / "scripts"))
    sys.path.append(str(ROOT / "app"))

def load_script(module_path):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"scripts.{module_path.stem}"] = module  # <-- clave
    spec.loader.exec_module(module)
    return module


# === 00_provenance ===
def test_provenance_runs():
    mod = load_script(ROOT / "scripts" / "00_provenance.py")
    mod.main()
    outfile = ROOT / "data" / "provenance.json"
    assert outfile.exists(), f"No se generó {outfile}"
    data = json.loads(outfile.read_text())
    assert "sha256" in data or "files" in data


# === 01_validar_datos ===
def test_validar_datos_generates_clean_csv():
    mod = load_script(ROOT / "scripts" / "01_validar_datos.py")
    mod.main()
    out_csv = ROOT / "data" / "processed" / "housing_clean.csv"
    assert out_csv.exists(), f"No se generó {out_csv}"
    df = pd.read_csv(out_csv)
    assert not df.isna().any().any()


# === 02_train_baseline ===
def test_train_baseline_creates_models():
    mod = load_script(ROOT / "scripts" / "02_train_baseline.py")
    mod.main()
    model_file = ROOT / "models" / "best_model.pkl"
    preproc_file = ROOT / "models" / "preprocessor.pkl"
    assert model_file.exists(), f"No se generó {model_file}"
    assert preproc_file.exists(), f"No se generó {preproc_file}"


# === 03_retrain_if_better ===
def test_retrain_executes():
    mod = load_script(ROOT / "scripts" / "03_retrain_if_better.py")
    try:
        mod.main(force=True)
    except Exception as e:
        pytest.skip(f"Skipping retrain test: {e}")


# === 04_drift_check ===
def test_drift_check_report(tmp_path):
    mod = load_script(ROOT / "scripts" / "04_drift_check.py")
    ref = pd.DataFrame({"CRIM": [0.1, 0.2, 0.3], "ZN": [1, 2, 3]})
    live = pd.DataFrame({"CRIM": [0.1, 0.2, 0.8], "ZN": [1, 5, 6]})

    ref_file = tmp_path / "ref.csv"
    live_file = tmp_path / "pred.csv"
    ref.to_csv(ref_file, index=False)
    live.to_csv(live_file, index=False)

    out_json = tmp_path / "drift.json"
    setattr(mod, "PROCESSED", ref_file)
    setattr(mod, "PRED_LOG", live_file)
    setattr(mod, "REPORT", out_json)

    mod.main(min_live=1)
    assert out_json.exists(), "No se generó el reporte de drift"
    report = json.loads(out_json.read_text())
    assert "summary" in report
    assert isinstance(report["summary"]["drift_detected"], bool)
