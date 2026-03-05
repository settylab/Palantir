import pytest
import pandas as pd
import numpy as np

from palantir.core import run_palantir, _compute_absorption_probabilities, _construct_markov_chain
from palantir.presults import PResults
from palantir.utils import optimize_absorption_rates


@pytest.fixture
def mock_data():
    np.random.seed(42)
    n_cells = 50
    n_genes = 10
    return pd.DataFrame(
        np.random.rand(n_cells, n_genes),
        columns=[f"gene_{i}" for i in range(n_genes)],
        index=[f"cell_{i}" for i in range(n_cells)],
    )


FILTER_WARNINGS = [
    pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated."),
    pytest.mark.filterwarnings(
        "ignore:Changing the sparsity structure of a csr_matrix is expensive."
    ),
]


@pytest.mark.parametrize("_", [None], ids=[""])
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
class TestAbsorptionRates:
    """Tests for partial absorption rate support."""

    def test_default_absorption_matches_original(self, _, mock_data):
        """absorption_rates=None should give the same result as the original code."""
        result_default = run_palantir(
            mock_data, "cell_0", terminal_states=["cell_1", "cell_2"]
        )
        result_explicit = run_palantir(
            mock_data, "cell_0", terminal_states=["cell_1", "cell_2"],
            absorption_rates={"cell_1": 1.0, "cell_2": 1.0},
        )
        pd.testing.assert_frame_equal(
            result_default.branch_probs, result_explicit.branch_probs, atol=1e-10
        )

    def test_partial_absorption_changes_fate_probs(self, _, mock_data):
        """Lowering absorption rate of one terminal should shift fate probs."""
        result_full = run_palantir(
            mock_data, "cell_0", terminal_states=["cell_1", "cell_2"]
        )
        result_partial = run_palantir(
            mock_data, "cell_0", terminal_states=["cell_1", "cell_2"],
            absorption_rates={"cell_1": 0.1, "cell_2": 1.0},
        )
        # Lowering cell_1's absorption rate should decrease its fate probability
        # relative to cell_2 for most cells
        assert not np.allclose(
            result_full.branch_probs.values, result_partial.branch_probs.values
        ), "Partial absorption should change fate probabilities"

    def test_absorption_rates_sum_to_one(self, _, mock_data):
        """Fate probabilities should still sum to ~1 for each cell."""
        result = run_palantir(
            mock_data, "cell_0", terminal_states=["cell_1", "cell_2"],
            absorption_rates={"cell_1": 0.3, "cell_2": 0.7},
        )
        row_sums = result.branch_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.05)

    def test_invalid_absorption_rate_raises(self, _, mock_data):
        """Absorption rates outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="Absorption rates must be in"):
            run_palantir(
                mock_data, "cell_0", terminal_states=["cell_1", "cell_2"],
                absorption_rates={"cell_1": 0.0, "cell_2": 1.0},
            )

    def test_returns_presults(self, _, mock_data):
        """Should return a PResults object even with partial absorption."""
        result = run_palantir(
            mock_data, "cell_0", terminal_states=["cell_1", "cell_2"],
            absorption_rates={"cell_1": 0.5, "cell_2": 0.5},
        )
        assert isinstance(result, PResults)


@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated.")
@pytest.mark.filterwarnings("ignore:Changing the sparsity structure of a csr_matrix is expensive.")
def test_optimize_absorption_rates(mock_data):
    """Test that optimization produces valid absorption rates."""
    result = optimize_absorption_rates(
        mock_data, "cell_0",
        terminal_states=["cell_1", "cell_2"],
        target_fate_probs=np.array([0.5, 0.5]),
        num_waypoints=50,
    )
    assert isinstance(result, dict)
    assert len(result) == 2
    for cell, rate in result.items():
        assert 0 < rate <= 1.0, f"Rate {rate} for {cell} out of bounds"
