"""remap_path_prefix relocates stored absolute paths without touching anything else."""

from pipeline.data_loader import remap_path_prefix

OLD = "/net/projects2/promega/2026_04_15_data"
NEW = "/local/data"


def test_remap_nested_structure_and_count() -> None:
    rec = {
        "a": {"cm_image_abs": f"{OLD}/x.png", "cm_image": "x.png"},
        "b": [f"{OLD}/y.png", "other/z.png", 3],
        "c": None,
    }
    out, n = remap_path_prefix(rec, OLD, NEW)
    assert n == 2
    assert out["a"]["cm_image_abs"] == f"{NEW}/x.png"
    assert out["a"]["cm_image"] == "x.png"
    assert out["b"] == [f"{NEW}/y.png", "other/z.png", 3]
    assert out["c"] is None
    assert set(out) == set(rec)


def test_prefix_only_at_start() -> None:
    out, n = remap_path_prefix(f"keep/{OLD}/x", OLD, NEW)
    assert n == 0 and out == f"keep/{OLD}/x"
