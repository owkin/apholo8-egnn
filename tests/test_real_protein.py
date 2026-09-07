"""Integration test: run end-to-end inference on a real protein PDB file.

The ESM call is mocked with pre-computed embeddings from embeddings.npz so the
test runs without downloading the 3B model.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from predictor import VNEGNNPredictor

# ---------------------------------------------------------------------------
# Paths — tests are skipped when the files are not present
# ---------------------------------------------------------------------------

_PROTEIN_DIR = Path(
    "/home/sagemaker-user/custom-file-systems/efs/"
    "fs-09913c1f7db79b6fd/laudmt/holo_data/data/coach420/raw/1a26A"
)
_SAFETENSORS = Path(__file__).parent.parent / "model.safetensors"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def esm_embeddings():
    return np.load(_PROTEIN_DIR / "embeddings.npz")["residue_embeddings"]


@pytest.fixture(scope="module")
def predictor():
    return VNEGNNPredictor(_SAFETENSORS, device="cpu")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shapes(predictor, esm_embeddings):
    """predict() must return arrays of shape [8, 3] and [8]."""
    with patch.object(predictor, "_compute_esm", return_value=esm_embeddings):
        coords, conf = predictor.predict(_PROTEIN_DIR / "protein.pdb")
    assert coords.shape == (8, 3)
    assert conf.shape == (8,)


def test_coords_are_finite(predictor, esm_embeddings):
    """Predicted binding-site coordinates must all be finite numbers."""
    with patch.object(predictor, "_compute_esm", return_value=esm_embeddings):
        coords, _ = predictor.predict(_PROTEIN_DIR / "protein.pdb")
    assert np.all(np.isfinite(coords)), "Predicted coordinates contain NaN or Inf"


def test_confidence_is_finite(predictor, esm_embeddings):
    """Confidence scores must all be finite numbers."""
    with patch.object(predictor, "_compute_esm", return_value=esm_embeddings):
        _, conf = predictor.predict(_PROTEIN_DIR / "protein.pdb")
    assert np.all(np.isfinite(conf)), "Confidence scores contain NaN or Inf"


def test_confidence_values_differ(predictor, esm_embeddings):
    """Virtual nodes must not all collapse to the same confidence score."""
    with patch.object(predictor, "_compute_esm", return_value=esm_embeddings):
        _, conf = predictor.predict(_PROTEIN_DIR / "protein.pdb")
    assert conf.std() > 0, "All confidence scores are identical — model may have collapsed"


def test_coords_within_protein_bounds(predictor, esm_embeddings):
    """Predicted site centers should lie within a generous radius of the protein centroid."""
    with patch.object(predictor, "_compute_esm", return_value=esm_embeddings):
        coords, _ = predictor.predict(_PROTEIN_DIR / "protein.pdb")
    binding = np.load(_PROTEIN_DIR / "binding.npz")
    centroid = binding["res_coords"].mean(axis=0)
    radius = np.linalg.norm(binding["res_coords"] - centroid, axis=1).max()
    distances = np.linalg.norm(coords - centroid, axis=1)
    assert np.all(distances < radius * 3), (
        "Some predicted sites are implausibly far from the protein"
    )
