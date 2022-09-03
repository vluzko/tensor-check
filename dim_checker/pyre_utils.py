"""Tools for working with Pyre"""
import json
import subprocess

from pathlib import Path
from mypy_extensions import TypedDict
from typing import Dict, List, Tuple


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
    # Kill existing server
    subprocess.run(['pyre', 'kill'], capture_output=True)
    subprocess.run(['pyre', 'start'], capture_output=True)
    cmd_args = ["pyre", "--noninteractive", "query", f"types(path='{str(path)}')"]
    process = subprocess.run(cmd_args, capture_output=True)
    stdout, stderr, return_code = process.stdout.decode(), process.stderr.decode(), process.returncode
    # pyre kill won't actually kill all of the executables and this uses a ton of RAM that won't get freed when the checker
    # finishes. Thanks Facebook.
    subprocess.run(['killall', 'pyre'], capture_output=True)
    subprocess.run(['killall', 'pyre.bin'], capture_output=True)
    resp: List[PyreResponse] = json.loads(stdout)["response"]
    return {x['path']: x['types'] for x in resp}


def pyre_location_to_tuple(x: PyreAnnotation) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    loc = x['location']
    start = loc['start']['line'], loc['start']['column']
    end = loc['end']['line'], loc['end']['column']
    return start, end
