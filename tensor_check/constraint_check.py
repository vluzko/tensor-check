from typing import Any
import z3
from tensor_check import checker, tc_types


def run_z3(context: checker.Context):
    all_types = context.get_all_types(0)

    for n in all_types:
        z3.Int(n)

    solver = z3.Solver()
    variables = {}

    for n, t in all_types.items():
        variables[n] = z3.Int(n)
        new_constraints = z3_constraint(t, variables)
        for c in new_constraints:
            solver.add(c)

    return solver.check()


def z3_constraint(t: tc_types.ChkType, variables: dict[str, Any]) -> list[Any]:
    """Generate constraints from this type."""
    constraints = []
    for constraint in t.constraints:
        match constraint:
            case tc_types.Equal(lhs, rhs):
                if lhs == tc_types.Self():

                    pass
    return constraints
