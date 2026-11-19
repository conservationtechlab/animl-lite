"""
Integration tests that run the real MiewID and MegaDescriptor ONNX models
stored in tests/fixtures. Tests are skipped when the model files (which are
gitignored) or a real onnxruntime install are not available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from animl import reid

FIXTURES = Path(__file__).parent / "fixtures"

MODELS = {
    "miewid": {
        "path": FIXTURES / "miewid_v3.onnx",
        "size": reid.MIEWID_SIZE,
        "normalize": reid.MIEWID_NORMALIZE,
        "dim": 2152,
        "dynamic_batch": False,
    },
    "megadescriptor_l": {
        "path": FIXTURES / "megadescriptor_l_384.onnx",
        "size": reid.MEGADESCRIPTOR_L_SIZE,
        "normalize": reid.MEGADESCRIPTOR_L_NORMALIZE,
        "dim": 1536,
        "dynamic_batch": True,
    },
}


def _real_onnxruntime() -> bool:
    import onnxruntime as ort
    # conftest installs a stub module when onnxruntime is missing
    return hasattr(ort, "__version__")


pytestmark = pytest.mark.skipif(not _real_onnxruntime(), reason="onnxruntime not installed")


@pytest.fixture(scope="module", params=list(MODELS))
def model_spec(request):
    name = request.param
    spec = MODELS[name]
    if not spec["path"].is_file():
        pytest.skip(f"model fixture missing: {spec['path']}")
    model = reid.load_reid(str(spec["path"]), device="cpu")
    return {"name": name, "model": model, **spec}


@pytest.fixture
def reid_manifest(tmp_path: Path) -> pd.DataFrame:
    """Three images: two identical patterned images and one distinct one."""
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    other = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    other[:, :80] = (200, 30, 30)

    paths = []
    for name, arr in [("a.jpg", pattern), ("a_copy.jpg", pattern), ("b.jpg", other)]:
        p = tmp_path / name
        Image.fromarray(arr).save(p, quality=95)
        paths.append(str(p))

    return pd.DataFrame({
        "filepath": paths,
        "frame": 0,
        "bbox_x": 0.0,
        "bbox_y": 0.0,
        "bbox_w": 1.0,
        "bbox_h": 1.0,
    })


def _extract(spec, manifest, **kwargs):
    kwargs.setdefault("num_workers", 0)
    return reid.extract_reid_embeddings(spec["model"],
                                        manifest,
                                        resize=spec["size"],
                                        normalize=spec["normalize"],
                                        **kwargs)


def test_load_reid_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        reid.load_reid(str(tmp_path / "missing.onnx"))


def test_load_reid_input_shape(model_spec):
    model = model_spec["model"]
    shape = model.get_inputs()[0].shape
    assert shape[1:] == [3, model_spec["size"], model_spec["size"]]


def test_extract_embeddings_shape_and_values(model_spec, reid_manifest):
    emb = _extract(model_spec, reid_manifest)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (len(reid_manifest), model_spec["dim"])
    assert np.all(np.isfinite(emb))
    # embeddings should not be degenerate
    assert np.all(np.linalg.norm(emb, axis=1) > 0)


def test_extract_embeddings_deterministic(model_spec, reid_manifest):
    first = _extract(model_spec, reid_manifest)
    second = _extract(model_spec, reid_manifest)
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-5)


def test_identical_images_closer_than_different(model_spec, reid_manifest):
    emb = _extract(model_spec, reid_manifest)
    dist = reid.compute_distance_matrix(emb, emb, metric="cosine")
    assert dist.shape == (3, 3)
    np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-5)
    # a.jpg and a_copy.jpg are the same image
    assert dist[0, 1] < 1e-4
    assert dist[0, 2] > dist[0, 1]
    np.testing.assert_allclose(dist, dist.T, atol=1e-6)


def test_extract_embeddings_with_prefetch_workers(model_spec, reid_manifest):
    single = _extract(model_spec, reid_manifest, num_workers=0)
    threaded = _extract(model_spec, reid_manifest, num_workers=2)
    np.testing.assert_allclose(single, threaded, rtol=1e-5, atol=1e-5)


def test_extract_embeddings_batched_matches_single(model_spec, reid_manifest):
    if not model_spec["dynamic_batch"]:
        pytest.skip(f"{model_spec['name']} export has a fixed batch size of 1")
    single = _extract(model_spec, reid_manifest, batch_size=1)
    batched = _extract(model_spec, reid_manifest, batch_size=2)
    assert batched.shape == single.shape
    np.testing.assert_allclose(single, batched, rtol=1e-3, atol=1e-4)


def test_extract_embeddings_uses_bbox_crop(model_spec, reid_manifest):
    full = _extract(model_spec, reid_manifest.iloc[[2]])
    cropped_manifest = reid_manifest.iloc[[2]].assign(bbox_w=0.5)
    cropped = _extract(model_spec, cropped_manifest)
    assert not np.allclose(full, cropped)


def test_extract_embeddings_custom_file_col(model_spec, reid_manifest):
    renamed = reid_manifest.rename(columns={"filepath": "path"})
    emb = _extract(model_spec, renamed, file_col="path")
    assert emb.shape == (len(reid_manifest), model_spec["dim"])


def test_extract_embeddings_missing_file_col(model_spec, reid_manifest):
    with pytest.raises(ValueError):
        _extract(model_spec, reid_manifest.drop(columns="filepath"))
