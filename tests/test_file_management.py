from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from animl import file_management
from animl.generator import ManifestGenerator, manifest_dataloader


def test_valid_extensions_contains_expected_types():
    assert ".jpg" in file_management.IMAGE_EXTENSIONS
    assert ".mp4" in file_management.VIDEO_EXTENSIONS
    assert file_management.IMAGE_EXTENSIONS.issubset(file_management.VALID_EXTENSIONS)
    assert file_management.VIDEO_EXTENSIONS.issubset(file_management.VALID_EXTENSIONS)


def test_build_file_manifest_finds_supported_files(image_dir: Path):
    manifest = file_management.build_file_manifest(image_dir, exif=False)
    assert not manifest.empty
    assert set(manifest["extension"]).issubset(file_management.VALID_EXTENSIONS)
    assert "notes.txt" not in manifest["filename"].tolist()


def test_build_file_manifest_nonrecursive_depth0(image_dir: Path):
    top_image = image_dir / "top.jpg"
    top_image.write_bytes(b"fake")
    manifest = file_management.build_file_manifest(image_dir, exif=False, recursive=False)
    assert len(manifest) == 1
    assert manifest.iloc[0]["filename"] == "top.jpg"


def test_build_file_manifest_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        file_management.build_file_manifest(tmp_path / "missing", exif=False)


def test_build_file_manifest_station_camera_columns(image_dir: Path):
    manifest = file_management.build_file_manifest(
        image_dir,
        exif=False,
        station_depth=1,
        camera_depth=2,
    )
    assert "station" in manifest.columns
    assert "camera" in manifest.columns
    assert set(manifest["station"].dropna()) == {"station_a", "station_b"}


@pytest.mark.parametrize("depth_key", ["station_depth", "camera_depth"])
def test_build_file_manifest_depth_validation_for_nonrecursive(image_dir: Path, depth_key: str):
    (image_dir / "top.jpg").write_text("x")
    kwargs = {"exif": False, "recursive": False, depth_key: 1}
    with pytest.raises(ValueError):
        file_management.build_file_manifest(image_dir, **kwargs)


def test_build_file_manifest_empty_dir_returns_empty_dataframe(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    manifest = file_management.build_file_manifest(empty, exif=False)
    assert isinstance(manifest, pd.DataFrame)
    assert manifest.empty


def test_working_directory_creates_expected_paths(tmp_path: Path):
    wd = file_management.WorkingDirectory(tmp_path)
    assert wd.basedir.is_dir()
    assert wd.filemanifest.name == "FileManifest.csv"
    assert wd.results.name == "Results.csv"


def test_working_directory_activate_dirs(tmp_path: Path):
    wd = file_management.WorkingDirectory(tmp_path)
    wd.activate_linkdir()
    wd.activate_visdir()
    assert wd.linkdir.is_dir()
    assert wd.visdir.is_dir()


def test_working_directory_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        file_management.WorkingDirectory(tmp_path / "nope")


def test_save_and_load_data_roundtrip(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = tmp_path / "out.csv"
    file_management.save_data(df, out, prompt=False)
    loaded = file_management.load_data(out)
    pd.testing.assert_frame_equal(loaded, df)


def test_load_data_requires_csv(tmp_path: Path):
    file = tmp_path / "data.txt"
    file.write_text("x")
    with pytest.raises(AssertionError):
        file_management.load_data(file)


def test_save_and_load_json_roundtrip(tmp_path: Path):
    payload = {"a": 1, "b": [1, 2]}
    out = tmp_path / "out.json"
    file_management.save_json(payload, out, prompt=False)
    loaded = file_management.load_json(out)
    assert loaded == payload


def test_load_json_requires_json(tmp_path: Path):
    file = tmp_path / "x.csv"
    file.write_text("a,b\n1,2")
    with pytest.raises(AssertionError):
        file_management.load_json(file)


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_save_and_load_yaml_roundtrip(tmp_path: Path, suffix: str):
    payload = {"name": "animl", "n": 2}
    out = tmp_path / f"out{suffix}"
    file_management.save_yaml(payload, out, prompt=False)
    loaded = file_management.load_yaml(out)
    assert loaded == payload


def test_load_yaml_requires_yaml(tmp_path: Path):
    file = tmp_path / "x.txt"
    file.write_text("name: animl")
    with pytest.raises(AssertionError):
        file_management.load_yaml(file)


@pytest.mark.parametrize(
    "response,expected",
    [
        ("y", True),
        ("n", False),
        ("invalid", False),
    ],
)
def test_check_file_prompt_responses(monkeypatch, tmp_path: Path, response: str, expected: bool):
    f = tmp_path / "file.csv"
    f.write_text("a,b\n1,2")
    monkeypatch.setattr("builtins.input", lambda _prompt: response)
    assert file_management.check_file(f, output_type="Manifest") is expected


def test_check_file_false_when_missing(tmp_path: Path):
    assert file_management.check_file(tmp_path / "missing.csv") is False


def test_class_list_to_dict_happy_path():
    df = pd.DataFrame({"id": [1, 2], "class": ["jaguar", "ocelot"]})
    out = file_management.class_list_to_dict(df)
    assert out == {1: "jaguar", 2: "ocelot"}


def test_class_list_to_dict_missing_columns_raises():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError):
        file_management.class_list_to_dict(df)


def test_active_times_groups_by_camera(tmp_path: Path):
    p1 = tmp_path / "c1" / "a.jpg"
    p1.parent.mkdir(parents=True)
    p1.write_text("x")
    p2 = tmp_path / "c1" / "b.jpg"
    p2.write_text("x")
    manifest = pd.DataFrame(
        {
            "filepath": [str(p1), str(p2)],
            "datetime": ["2026-01-01 00:00:00", "2026-01-01 00:01:00"],
            "camera": ["c1", "c1"],
        }
    )
    times = file_management.active_times(manifest)
    assert ("datetime", "min") in times.columns
    assert ("datetime", "max") in times.columns


def test_active_times_adds_timestamp_when_missing(tmp_path: Path):
    p = tmp_path / "camx" / "a.jpg"
    p.parent.mkdir(parents=True)
    p.write_text("x")
    manifest = pd.DataFrame({"filepath": [str(p)]})
    times = file_management.active_times(manifest, camera_depth=0)
    assert len(times) == 1


def test_active_times_validates_manifest_type():
    with pytest.raises(ValueError):
        file_management.active_times(["not", "df"])


def test_sequence_calculation_assigns_sequences():
    manifest = pd.DataFrame(
        {
            "filepath": ["a.jpg", "b.jpg", "c.jpg"],
            "station": ["s1", "s1", "s1"],
            "datetime": ["2026-01-01 00:00:00", "2026-01-01 00:00:30", "2026-01-01 00:03:00"],
        }
    )
    out = file_management.sequence_calculation(manifest, station_col="station", maxdiff=60)
    assert out["sequence"].tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "station_col,maxdiff,exc",
    [
        ("", 60, Exception),
        ("station", -1, Exception),
    ],
)
def test_sequence_calculation_input_validation(station_col, maxdiff, exc):
    manifest = pd.DataFrame({"filepath": ["a.jpg"], "station": ["s"], "datetime": ["2026-01-01 00:00:00"]})
    with pytest.raises(exc):
        file_management.sequence_calculation(manifest, station_col=station_col, maxdiff=maxdiff)


def test_manifest_generator_len_and_item(tmp_path: Path):
    img = tmp_path / "img.jpg"
    from PIL import Image

    Image.new("RGB", (20, 10), color=(100, 50, 10)).save(img)
    df = pd.DataFrame({"filepath": [str(img)]})
    gen = ManifestGenerator(df, crop=False, resize_height=16, resize_width=16)
    assert len(gen) == 1
    item = gen[0]
    assert item is not None
    img_arr, path, frame, hw = item
    assert img_arr.shape == (3, 16, 16)
    assert path == str(img)
    assert frame == 0
    assert hw.tolist() == [10, 20]


def test_manifest_generator_requires_bbox_when_crop_true(tmp_path: Path):
    img = tmp_path / "img.jpg"
    from PIL import Image

    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(img)
    df = pd.DataFrame({"filepath": [str(img)]})
    with pytest.raises(ValueError):
        ManifestGenerator(df, crop=True)


def test_manifest_generator_invalid_crop_coord(tmp_path: Path):
    img = tmp_path / "img.jpg"
    from PIL import Image

    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(img)
    df = pd.DataFrame({"filepath": [str(img)], "bbox_x": [0], "bbox_y": [0], "bbox_w": [1], "bbox_h": [1]})
    with pytest.raises(ValueError):
        ManifestGenerator(df, crop=True, crop_coord="bad")


def test_manifest_dataloader_yields_batches(tmp_path: Path):
    img = tmp_path / "img.jpg"
    from PIL import Image

    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(img)
    df = pd.DataFrame({"filepath": [str(img)]})
    loader = manifest_dataloader(df, crop=False, resize_height=8, resize_width=8)
    batch = next(loader)
    assert batch[0].shape == (1, 3, 8, 8)
    assert batch[1] == [str(img)]
