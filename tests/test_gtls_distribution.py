from importlib.metadata import distribution
from pathlib import Path


def test_gtls_comes_from_pinned_pypi_distribution():
    """Require the supported GTLS version to resolve outside this checkout."""
    gtls_distribution = distribution("gputls")

    assert gtls_distribution.version == "0.6.1"
    package_root = Path(gtls_distribution.locate_file("gputls")).resolve()
    repository_root = Path(__file__).resolve().parents[1]
    assert package_root.is_dir()
    assert not package_root.is_relative_to(repository_root)
