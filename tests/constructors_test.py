import pytest
from tests import TEST_FOLDER
from dim_checker import cst_checker


def test_ones():
    with pytest.raises(AssertionError):
        cst_checker.check_file(TEST_FOLDER / 'construct.py')
