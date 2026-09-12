"""Explicit dependency on audited Q3 types, protocol and fixed error field.

Q3 Channel.update and Q3 world feedback are NOT used for Q4 observations.
"""
import importlib
import importlib.util
from pathlib import Path
import sys

SHARED_DIR = Path(__file__).resolve().parents[1] / 'vendor' / 'q3'
PACKAGE = '_q4_shared_q3'
if PACKAGE not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE, SHARED_DIR / '__init__.py', submodule_search_locations=[str(SHARED_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)


def load(name):
    return importlib.import_module(f'{PACKAGE}.{name}')


core = load('core')
Action, Observation, GeometryError = core.Action, core.Observation, core.GeometryError
disk, distance, point_key = core.disk, core.distance, core.point_key
