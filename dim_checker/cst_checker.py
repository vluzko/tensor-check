import libcst as cst

from pathlib import Path
from libcst._position import CodeRange
from libcst.metadata import PositionProvider
from typing import Any, Dict, List, Optional, Tuple

from dim_checker.types import ChkType, Module, NoneType
from dim_checker import pyre_utils, torch_annotations


class Scope:
    names: Dict[str, ChkType]

    def __init__(self):
        self.names = {"torch": Module({"arange": NoneType})}

    def __contains__(self, key: str) -> bool:
        return key in self.names

    def __getitem__(self, key: str):
        return self.names[key]


class Context:
    scopes: List[Scope]
    imports: Any

    def __init__(self):
        self.scopes = [Scope()]

    def new_scope(self) -> int:
        self.scopes.append(Scope())
        return len(self.scopes)

    def has_name(self, name: str, scope_id: int = 0) -> bool:
        return name in self.scopes[scope_id]

    def lookup_name(self, name: str, scope_id: int = 0) -> ChkType:
        scope = self.scopes[scope_id]
        return scope[name]

    def add_type(self, name: str, node_type: ChkType, scope_id: int = 0):
        self.scopes[scope_id].names[name] = node_type

    def lookup_node(self, node) -> ChkType:
        return NoneType()


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
        self.types = {}  # type: ignore
        self.ctx = Context()

    def visit_Attribute(self, node: cst.Attribute):
        node.value.visit(self)
        base_node_type = self.ctx.lookup_node(node.value)
        # node_type = self.ctx.lookup_name(node.value.value)
        # TODO: Lookup types
        pass

    def visit_Import(self, node: cst.Import) -> bool:
        if node.names[0].name.value == "torch":
            self.ctx.add_type("torch", torch_annotations.TorchType)
        return False

    def visit_Name(self, node: cst.Name) -> None:
        # if node.value == 'arange':
        #     import pdb
        #     pdb.set_trace()
        # Only print out names that are parameters
        # if self.get_metadata(PositionProvider, node):
        position = code_range_to_tuple(self.get_metadata(PositionProvider, node))
        if position in self.by_position:
            print(node)
            cached_type = self.by_position[position]
            print(cached_type)
            self.types[node] = cached_type
        else:
            pass

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> None:
        # print(node)
        pass


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
        print(checker.types_cache)


def check_code(code: str):
    raise NotImplementedError


if __name__ == "__main__":
    # Pyre is extremely finicky about the paths you pass to it: they have to match a particular syntax
    code_path = Path("tests")
    paths = ["bin_op.py"]
    # f = (code_path / paths[0]).open().read()
    # cache = TypeInferenceProvider.gen_cache(Path(code_path), paths, None)
    check_file(code_path / paths[0])
