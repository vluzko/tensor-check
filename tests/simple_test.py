"""Basic tests that the checker runs correctly."""
from pathlib import Path
from dim_checker import checker

TEST_FOLDER = Path(__file__).parent


def test_bin_op():
    f = TEST_FOLDER / "bin_op.py"
    contents = f.open().read()
    checker.check(contents)


def test_def_and_call():
    f = TEST_FOLDER / "def_and_call.py"
    contents = f.open().read()
    checker.check(contents)


def test_class_and_obj():
    f = TEST_FOLDER / "class_and_obj.py"
    contents = f.open().read()
    checker.check(contents)


def test_module():
    f = TEST_FOLDER / "example_nn_module.py"
    contents = f.open().read()
    checker.check(contents)
