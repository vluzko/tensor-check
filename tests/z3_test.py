from pathlib import Path
from tensor_check import checker, tc_types, constraint_check

TEST_FOLDER = Path(__file__).parent / "test_files"


def test_integers():
    f = TEST_FOLDER / "arithmetic.py"
    context = checker.check_file(f)
    constraint_check.run_z3(context)
