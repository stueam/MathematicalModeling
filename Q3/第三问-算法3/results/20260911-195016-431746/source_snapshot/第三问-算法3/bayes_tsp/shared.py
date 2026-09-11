"""Reuse the repository's audited geometry/protocol without importing its MC.

The sibling directory is an explicit delivery dependency. A private package
namespace prevents collisions with other projects called q3; no sys.path edits.
"""
import importlib
import importlib.util
from pathlib import Path
import sys


SHARED_DIR = Path(__file__).resolve().parents[2] / '第三问-算法1' / 'q3'
PACKAGE = '_q3_bayes_shared'
if PACKAGE not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE, SHARED_DIR / '__init__.py', submodule_search_locations=[str(SHARED_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)


def load(name):
    return importlib.import_module(f'{PACKAGE}.{name}')


core = load('core')
Action, Belief, Observation = core.Action, core.Belief, core.Observation
GeometryError = core.GeometryError
disk, distance, point_key = core.disk, core.distance, core.point_key
Baseline = load('policy').Baseline
