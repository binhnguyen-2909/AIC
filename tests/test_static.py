import tempfile
from pathlib import Path
from unittest.mock import patch

try:
    from solution import qa_vlm
except (ImportError, ModuleNotFoundError):
    qa_vlm = None


ROOT = Path(__file__).resolve().parents[1]


def test_repository_has_no_dataset_extensions():
    forbidden = {".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".npy", ".faiss", ".zip"}
    found = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in forbidden]
    assert not found, f"dataset/generated files found: {found[:5]}"


def test_production_entrypoint_exists():
    assert (ROOT / "solution" / "submission_ens.py").is_file()
    assert (ROOT / "solution" / "ensemble.py").is_file()


def test_qa_context_mapping_is_bounded_and_temporally_ordered():
    if qa_vlm is None:
        return
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        map_dir = root / "extracted" / "map-keyframes-aic25-b1" / "map-keyframes"
        map_dir.mkdir(parents=True)
        (map_dir / "L01_V001.csv").write_text(
            "n,pts_time,fps,frame_idx\n"
            "1,0,1,10\n2,1,1,20\n3,2,1,30\n4,3,1,40\n",
            encoding="utf-8",
        )
        keyframe_dir = root / "extracted" / "Keyframes_L01" / "keyframes" / "L01_V001"
        keyframe_dir.mkdir(parents=True)
        for ordinal in range(1, 5):
            (keyframe_dir / f"{ordinal:03d}.jpg").write_bytes(b"placeholder")

        with patch.object(qa_vlm, "ROOT", root), patch.object(
            qa_vlm, "KEYFRAMES_BASE", root / "extracted"
        ):
            paths = qa_vlm.keyframe_paths("L01_V001", 22, max_frames=3)

        assert [path.name for path in paths] == ["001.jpg", "002.jpg", "003.jpg"]
        assert len(paths) <= 3
