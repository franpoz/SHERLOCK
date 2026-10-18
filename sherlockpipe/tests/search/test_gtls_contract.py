import ast
from importlib.util import find_spec
from pathlib import Path

import pytest


def test_vendored_gtls_version_is_packaged():
    package_spec = find_spec("gputls")
    assert package_spec is not None
    version_path = Path(package_spec.origin).with_name("version.py")
    version_tree = ast.parse(version_path.read_text(encoding="utf-8"))
    version = next(
        ast.literal_eval(node.value)
        for node in version_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "GTLS_VERSION"
                for target in node.targets)
    )
    assert version == "0.5.1"


def test_vendored_public_api():
    pytest.importorskip("gputls")
    from gputls import gtls

    assert callable(gtls)
    assert hasattr(gtls, "power")
