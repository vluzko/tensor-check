from pathlib import Path
from dim_checker import checker

TEST_FOLDER = Path(__file__).parent


def test_bin_op():
    f = Path(__file__).parent / 'example_file.py'
    contents = f.open().read()
    checker.check(contents)


def test_constant():
    f = Path(__file__).parent / 'example_2.py'
    contents = f.open().read()
    checker.check(contents)


def test_module():
    f = TEST_FOLDER / 'example_nn_module.py'
    contents = f.open().read()
    checker.check(contents)