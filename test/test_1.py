from pathlib import Path
from dim_checker import checker


def test_1():
    f = Path(__file__).parent / 'example_file.py'
    contents = f.open().read()
    checker.check(contents)
