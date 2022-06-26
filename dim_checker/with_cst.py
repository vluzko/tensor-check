import json
import subprocess
from typing import List, Optional, Tuple
import libcst as cst
from pathlib import Path

from libcst._position import CodeRange
from libcst.metadata import PositionProvider
from libcst.metadata.type_inference_provider import PyreData, TypeInferenceProvider

# x = cst.parse_module("import torch")

# Pyre is extremely finicky about the paths you pass to it: they have to match a particular syntax
code_path = Path('test')
paths = ['bin_op.py']


def check_pyre_config():
    """Make sure Pyre paths are configured correctly
    Pyre is extremely finicky about paths. The paths you pass have to match the particular
    syntax used in the config file, not just point to the same files.

    """
    raise NotImplementedError


def run_command(cmd_args: List[str], timeout: Optional[int] = None) -> Tuple[str, str, int]:
    process = subprocess.run(cmd_args, capture_output=True, timeout=timeout)
    return process.stdout.decode(), process.stderr.decode(), process.returncode


def get_pyre_types(path: Path):
    cmd_args = ["pyre", "--noninteractive", "query", f"types(path='{str(path)}')"]
    try:
        stdout, stderr, return_code = run_command(cmd_args, timeout=None)
    except subprocess.TimeoutExpired as exc:

        raise exc

    if return_code != 0:
        raise Exception(f"stderr:\n {stderr}\nstdout:\n {stdout}")
    try:
        resp = json.loads(stdout)["response"]
    except Exception as e:
        raise Exception(f"{e}\n\nstderr:\n {stderr}\nstdout:\n {stdout}")
    import pdb
    pdb.set_trace()


def pyre_location_to_tuple(x: dict) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Convert the output of Pyre check to a tuple"""
    return tuple(x['location']['start'].values()), tuple(x['location']['stop'].values())  # type: ignore


def code_range_to_tuple(c: CodeRange) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Convert a CodeRange to a tuple"""
    return ((c.start.line, c.start.column), (c.end.line, c.end.column))


class Checker(cst.CSTVisitor):
    """Type checker
    The built-in TypeInferenceProvider is totally undocumented and I'm not wasting
    more time trying to get it to work, so I just pass the Pyre results directly
    """
    METADATA_DEPENDENCIES = (PositionProvider,)
    def __init__(self, cache) -> None:
        super().__init__()
        self.types_cache = cache
        self.by_position = {(tuple(x['location']['start'].values()), tuple(x['location']['stop'].values())): x for x in self.types_cache}

    def visit_Name(self, node: cst.Name) -> None:
        # Only print out names that are parameters
        if self.get_metadata(PositionProvider, node):
            position = code_range_to_tuple(self.get_metadata(PositionProvider, node))
            cached_type = self.by_position[position]
            print(cached_type)


f = (code_path / paths[0]).open().read()
module = cst.parse_module(f)
cache = TypeInferenceProvider.gen_cache(Path(code_path), paths, None)
wrapper = cst.MetadataWrapper(module)
result = wrapper.visit(Checker(cache['bin_op.py']['types']))  # type: ignore
# wrapper = cst.MetadataWrapper(module)
# print(wrapper)
# wrapper.resolve(TypeInferenceProvider)
# import pdb
# pdb.set_trace()
# wrapper = cst.MetadataWrapper(module, cache={TypeInferenceProvider: cache})
# result = wrapper.visit(Checker())