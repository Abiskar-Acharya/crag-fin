"""Sanity check that the package imports."""

def test_package_imports():
    import crag_fin
    assert crag_fin.__version__ == "0.1.0"


def test_subpackages_import():
    from crag_fin import (  # noqa: F401
        adversarial,
        auditor,
        baselines,
        calibrator,
        decomposer,
        eval,
        reasoner,
        retriever,
        tools,
    )
    # if any of these blow up, the __init__.py files have a problem
