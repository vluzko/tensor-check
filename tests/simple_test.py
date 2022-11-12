"""Basic tests that the checker runs correctly."""
from pathlib import Path
from tensor_check import checker

TEST_FOLDER = Path(__file__).parent / "test_files"


def test_const():
    f = TEST_FOLDER / "const.py"
    res = checker.check_file(f)


def test_bin_op():
    f = TEST_FOLDER / "bin_op.py"
    res = checker.check_file(f)


def test_def_and_call():
    f = TEST_FOLDER / "def_and_call.py"
    res = checker.check_file(f)


def test_class_and_obj():
    f = TEST_FOLDER / "class_and_obj.py"
    res = checker.check_file(f)


def test_module():
    f = TEST_FOLDER / "nn_module.py"
    res = checker.check_file(f)
