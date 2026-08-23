import numpy as np

from src.visual_train import projected_kcenter_indices


def test_projected_kcenter_is_deterministic_and_retains_original_vectors():
    rng = np.random.default_rng(7)
    memory = rng.normal(size=(40, 384)).astype(np.float32)
    first = projected_kcenter_indices(memory, retention_ratio=.05, projection_dim=16, seed=42, chunk_size=9)
    second = projected_kcenter_indices(memory, retention_ratio=.05, projection_dim=16, seed=42, chunk_size=9)
    assert np.array_equal(first, second)
    assert len(first) == 2 and len(np.unique(first)) == 2
    selected = memory[first]
    assert selected.shape == (2, 384)
    assert np.array_equal(selected, memory[first])


def test_projected_kcenter_first_selection_and_tie_behavior():
    memory = np.zeros((20, 384), dtype=np.float32)
    selected = projected_kcenter_indices(memory, retention_ratio=.1, projection_dim=16, seed=42)
    assert selected.tolist() == [0, 1]
