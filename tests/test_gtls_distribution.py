from importlib.metadata import distribution
from pathlib import Path


def test_gtls_comes_from_pinned_pypi_distribution():
    gtls_distribution = distribution("gputls")

    assert gtls_distribution.version == "0.6.1"
    distribution_root = Path(gtls_distribution.locate_file("")).resolve()
    repository_root = Path(__file__).resolve().parents[1]
    assert not distribution_root.is_relative_to(repository_root)
