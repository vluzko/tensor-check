from pathlib import Path
from dim_checker import checker


def test_1():
    f = Path(__file__).parent / 'example_file.py'
    contents = f.open().read()
    checker.check(contents)


def test_2():
    f = Path(__file__).parent / 'example_2.py'
    contents = f.open().read()
    checker.check(contents)