"""S21 initialization from the audited paired experiment; runtime rules unchanged."""
import json
from functools import lru_cache
from pathlib import Path
from check_s21_certificate import verify
from q4.core import DOMAIN
from q4.coverage import full_exclusion

@lru_cache(maxsize=1)
def stations():
    path=Path(__file__).parent/'s21_certificate.json'
    verify(path)
    points=tuple(tuple(map(float,p)) for p in json.loads(path.read_text())['info']['points'])
    if not DOMAIN.difference(full_exclusion(points)).is_empty:
        raise ValueError('S21 initialization coverage failed in this geometry environment')
    return points

def install():
    from q4 import sector
    stations()
    sector.ring_stations=stations
    return stations()
