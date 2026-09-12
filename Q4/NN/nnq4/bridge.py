"""Frozen audited Q4 geometry; no imports of a live sibling algorithm."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / '第四问'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from q4.core import Belief, DOMAIN
from q4.shared import Action, distance, point_key, GeometryError
from q4.sector import ProbePolicy, ring_stations
from q4.policy import Config
from q4.posterior import Model, QuadratureError
from q4.localization import guaranteed_clear, local_attempts, finite_clear
