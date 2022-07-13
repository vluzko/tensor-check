import json
import subprocess
from typing import Dict, List, Optional, Tuple
import libcst as cst
from pathlib import Path

from libcst._position import CodeRange
from libcst.metadata import PositionProvider
from libcst.metadata.type_inference_provider import PyreData, TypeInferenceProvider
from mypy_extensions import TypedDict


PyreLC = TypedDict('PyreLC', {'line': int, 'column': int})
PyreLocation = TypedDict('PyreLocation', {'start': PyreLC, 'end': PyreLC})
PyreAnnotation = TypedDict('PyreAnnotation', {'location': PyreLocation, 'annotation': str})
PyreResponse = TypedDict('PyreResponse', {'path': str, 'types': List[PyreAnnotation]})


def check_pyre_config():
    """Make sure Pyre paths are configured correctly
    Pyre is extremely finicky about paths. The paths you pass have to match the particular
    syntax used in the config file, not just point to the same files.

    """
    raise NotImplementedError


def get_pyre_types(path: Path) -> Dict[str, List[PyreAnnotation]]:
    """Get all Pyre type annotations for the given path."""
    cmd_args = ["pyre", "--noninteractive", "query", f"types(path='{str(path)}')"]
    process = subprocess.run(cmd_args, capture_output=True)
    stdout, stderr, return_code = process.stdout.decode(), process.stderr.decode(), process.returncode

    assert return_code == 0, f"stderr:\n {stderr}\nstdout:\n {stdout}"
    resp: List[PyreResponse] = json.loads(stdout)["response"]
    return {x['path']: x['types'] for x in resp}


def pyre_location_to_tuple(x: PyreAnnotation) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Convert the output of Pyre check to a tuple"""
    loc = x['location']
    start = loc['start']['line'], loc['start']['column']
    end = loc['end']['line'], loc['end']['column']
    return start, end


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


def check_file(path: Path):
    """Run the checker on a path."""
    if path.is_dir():
        raise NotImplementedError
    else:
        pyre_types = get_pyre_types(path)

        f = path.open().read()
        module = cst.parse_module(f)
        wrapper = cst.MetadataWrapper(module)
        checker = Checker(pyre_types[str(path)])
        wrapper.visit(checker)


if __name__ == '__main__':
    # Pyre is extremely finicky about the paths you pass to it: they have to match a particular syntax
    code_path = Path('test')
    paths = ['bin_op.py']
    # f = (code_path / paths[0]).open().read()
    # cache = TypeInferenceProvider.gen_cache(Path(code_path), paths, None)
    check_file(code_path / paths[0])