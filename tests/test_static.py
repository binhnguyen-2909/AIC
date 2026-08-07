from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repository_has_no_dataset_extensions():
    forbidden = {".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".npy", ".faiss", ".zip"}
    found = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in forbidden]
    assert not found, f"dataset/generated files found: {found[:5]}"


def test_production_entrypoint_exists():
    assert (ROOT / "solution" / "submission_ens.py").is_file()
    assert (ROOT / "solution" / "ensemble.py").is_file()
