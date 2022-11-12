import libcst as cst

from pathlib import Path
from libcst._position import CodeRange
from libcst.metadata import PositionProvider
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tensor_check import pyre_utils, tc_types, torch_annotations


class Scope:
    parent_scope: Optional["Scope"]
    names: Dict[str, cst.CSTNode]

    def __init__(self, parent: Optional["Scope"] = None):
        # TODO: Needs to be an ordered dict
        self.names = {}
        self.parent = parent

    def __contains__(self, key: str) -> bool:
        return key in self.names

    def __getitem__(self, key: str):
        return self.names[key]

    def __setitem__(self, key: str, node: cst.CSTNode):
        self.names[key] = node

    def all_names(self) -> Iterable[str]:
        if self.parent_scope is not None:
            return (*self.names.keys(), *self.parent_scope.all_names())
        else:
            return (x for x in self.names.keys())


class Context:
    scopes: List[Scope]
    types: Dict[cst.CSTNode, tc_types.ChkType]
    imports: Any

    def __init__(self):
        self.scopes = [Scope()]
        self.types = {}

    def new_scope(self) -> int:
        self.scopes.append(Scope())
        return len(self.scopes)

    def has_name(self, name: str, scope_id: int = 0) -> bool:
        return name in self.scopes[scope_id]

    def lookup_name(self, name: str, scope_id: int = 0) -> tc_types.ChkType:
        scope = self.scopes[scope_id]
        node = scope[name]
        return self.types[node]

    def add_type(
        self,
        node,
        node_type: tc_types.ChkType,
        name: Optional[str] = None,
        scope_id: int = 0,
    ):
        self.types[node] = node_type
        if name is not None:
            self.scopes[scope_id][name] = node

    def lookup_node(self, node) -> tc_types.ChkType:
        if isinstance(node, cst.Name):
            return self.lookup_name(node.value)
        else:
            return self.types[node]


class Checker(cst.CSTVisitor):
    """Type checker
    The built-in TypeInferenceProvider is totally undocumented and I'm not wasting
    more time trying to get it to work, so I just pass the Pyre results directly
    """

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, cache: List[pyre_utils.PyreAnnotation]) -> None:
        super().__init__()
        self.types_cache = cache
        self.by_position = {
            (
                tuple(x["location"]["start"].values()),
                tuple(x["location"]["stop"].values()),
            ): x
            for x in self.types_cache
        }
        self.ctx = Context()

    def visit_Assign(self, node: cst.Assign):
        node.value.visit(self)

        # Assign the name in this context
        rhs_type = self.ctx.lookup_node(node.value)
        assert len(node.targets) == 1
        # TODO: Multiple targets
        # TODO: Other forms of assignment
        self.ctx.add_type(node.targets[0].target.value, rhs_type)

    def visit_Attribute(self, node: cst.Attribute):
        node.value.visit(self)
        base_node_type = self.ctx.lookup_node(node.value)
        assert isinstance(node.attr, cst.Name)
        attribute_type = base_node_type.attributes[node.attr.value]
        self.ctx.add_type(node, attribute_type)

    def visit_Call(self, node: cst.Call):
        node.func.visit(self)
        func_type = self.ctx.lookup_node(node.func)
        # TODO: Handle Args
        # TODO: Handle kwargs
        # TODO: Get return type
        import pdb

        pdb.set_trace()

    def visit_FunctionDef(self, node):
        # TODO: Get arg types
        # TODO: Visit body
        # TODO: Store function type
        pass

    def visit_Import(self, node: cst.Import):
        if node.names[0].name.value == "torch":
            self.ctx.add_type(node, torch_annotations.TorchType, "torch")

    def visit_Name(self, node: cst.Name) -> None:
        position = code_range_to_tuple(self.get_metadata(PositionProvider, node))
        if position in self.by_position:
            cached_type = self.by_position[position]
            # TODO: Convert pyre to internal type
            self.ctx.add_type(node, cached_type, node.value)
        else:
            pass

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> None:
        pass

    def visit_Integer(self, node: cst.Integer) -> None:
        t = tc_types.InternalInt()
        t.constraints = [tc_types.Equal(tc_types.Self(), node.value)]
        self.ctx.add_type(node, t)

    def visit_Float(self, node: cst.Float) -> None:
        t = tc_types.InternalFloat()
        t.constraints = [tc_types.Equal(tc_types.Self(), node.value)]
        self.ctx.add_type(node, t)


def check_pyre_config():
    """Make sure Pyre paths are configured correctly
    Pyre is extremely finicky about paths. The paths you pass have to match the particular
    syntax used in the config file, not just point to the same files.

    """
    raise NotImplementedError


def code_range_to_tuple(c: CodeRange) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Convert a CodeRange to a tuple"""
    return ((c.start.line, c.start.column), (c.end.line, c.end.column))


def check_file(path: Path):
    """Run the checker on a path."""
    if path.is_dir():
        raise NotImplementedError
    else:
        pyre_types = pyre_utils.get_pyre_types(path)
        f = path.open().read()
        module = cst.parse_module(f)
        wrapper = cst.MetadataWrapper(module)
        checker = Checker(pyre_types[str(path)])
        wrapper.visit(checker)
        print(checker.ctx)
        return checker.ctx


def check_code(code: str):
    raise NotImplementedError


if __name__ == "__main__":
    # Pyre is extremely finicky about the paths you pass to it: they have to match a particular syntax
    code_path = Path("tests") / "test_files"
    paths = ["bin_op.py"]
    # f = (code_path / paths[0]).open().read()
    # cache = TypeInferenceProvider.gen_cache(Path(code_path), paths, None)
    check_file(code_path / paths[0])
