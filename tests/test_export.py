from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from animl import export


def test_export_folders_creates_links(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_dir = tmp_path / "sorted"
    out_dir.mkdir()
    manifest = required_results_manifest.copy()
    out = export.export_folders(manifest, out_dir, label_col="prediction", copy=False)
    assert "link" in out.columns
    link = Path(out.loc[0, "link"])
    assert link.exists()


def test_export_folders_copy_mode(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_dir = tmp_path / "sorted_copy"
    out_dir.mkdir()
    manifest = required_results_manifest.copy()
    out = export.export_folders(manifest, out_dir, label_col="prediction", copy=True)
    link = Path(out.loc[0, "link"])
    assert link.exists()


def test_export_folders_requires_label_col(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_dir = tmp_path / "sorted_missing"
    out_dir.mkdir()
    manifest = required_results_manifest.drop(columns=["prediction"])
    with pytest.raises(AssertionError):
        export.export_folders(manifest, out_dir, label_col="prediction")


def test_remove_link_deletes_file_and_column(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_dir = tmp_path / "sorted_remove"
    out_dir.mkdir()
    manifest = export.export_folders(required_results_manifest.copy(), out_dir, label_col="prediction")
    link_path = Path(manifest.loc[0, "link"])
    assert link_path.exists()
    out = export.remove_link(manifest)
    assert "link" not in out.columns
    assert not link_path.exists()


def test_remove_link_requires_column(required_results_manifest: pd.DataFrame):
    with pytest.raises(AssertionError):
        export.remove_link(required_results_manifest.copy(), link_col="link")


def test_update_labels_from_folders_reads_label_from_parent_dir(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_dir = tmp_path / "review"
    out_dir.mkdir()
    sorted_manifest = export.export_folders(required_results_manifest.copy(), out_dir, label_col="prediction")
    updated = export.update_labels_from_folders(sorted_manifest, out_dir)
    assert "label" in updated.columns
    assert updated.loc[0, "label"] == "jaguar"


def test_export_coco_writes_expected_shape(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_file = tmp_path / "out_coco.json"
    class_dict = {7: "jaguar"}
    coco = export.export_coco(required_results_manifest.copy(), class_dict, out_file)
    assert out_file.exists()
    assert set(coco.keys()) == {"info", "licenses", "images", "annotations", "categories"}
    assert len(coco["images"]) == 1
    assert len(coco["categories"]) == 1
    assert coco["categories"][0]["id"] == 7


def test_export_coco_skips_nan_bbox(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_file = tmp_path / "out_coco_nan.json"
    manifest = required_results_manifest.copy()
    manifest.loc[0, "bbox_x"] = float("nan")
    coco = export.export_coco(manifest, {1: "jaguar"}, out_file)
    assert len(coco["annotations"]) == 0


@pytest.mark.parametrize(
    "missing_col",
    [
        "filepath",
        "filename",
        "filemodifydate",
        "frame",
        "max_detection_conf",
        "category",
        "conf",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "prediction",
        "confidence",
    ],
)
def test_export_coco_requires_all_columns(tmp_path: Path, required_results_manifest: pd.DataFrame, missing_col: str):
    out_file = tmp_path / "out.json"
    manifest = required_results_manifest.drop(columns=[missing_col])
    with pytest.raises(AssertionError):
        export.export_coco(manifest, {1: "jaguar"}, out_file)


def test_export_timelapse_animals_only(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_dir = tmp_path / "timelapse1"
    csv_loc = export.export_timelapse(required_results_manifest.copy(), out_dir, only_animal=True)
    assert csv_loc.exists()
    assert (out_dir / "animals.csv").exists()
    assert (out_dir / "manifest.csv").exists()


def test_export_timelapse_with_non_animals(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_dir = tmp_path / "timelapse2"
    manifest = pd.concat(
        [
            required_results_manifest.copy(),
            required_results_manifest.assign(category=2, category_label="human", prediction="human"),
        ],
        ignore_index=True,
    )
    csv_loc = export.export_timelapse(manifest, out_dir, only_animal=False)
    assert csv_loc.exists()
    assert (out_dir / "non-animals.csv").exists()


def test_export_megadetector_writes_expected_json(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_file = tmp_path / "md.json"
    md = export.export_megadetector(required_results_manifest.copy(), out_file=out_file, prompt=False)
    assert out_file.exists()
    assert "images" in md
    assert "detection_categories" in md
    assert md["images"][0]["detections"][0]["category"] == 1


def test_export_megadetector_skips_empty_rows(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_file = tmp_path / "md_empty.json"
    manifest = required_results_manifest.copy()
    manifest.loc[0, "category"] = 0
    md = export.export_megadetector(manifest, out_file=out_file, prompt=False)
    assert md["images"][0]["detections"] == []


def test_export_megadetector_requires_columns(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_file = tmp_path / "bad.json"
    manifest = required_results_manifest.drop(columns=["bbox_w"])
    with pytest.raises(ValueError):
        export.export_megadetector(manifest, out_file=out_file, prompt=False)


def test_export_megadetector_output_is_valid_json(tmp_path: Path, required_results_manifest: pd.DataFrame):
    out_file = tmp_path / "md2.json"
    export.export_megadetector(required_results_manifest.copy(), out_file=out_file, prompt=False)
    loaded = json.loads(out_file.read_text())
    assert loaded["info"]["format_version"] == "3.0"
