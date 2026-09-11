"""Algorithm two. Share the audited physical/protocol kernel with algorithm one."""
from pathlib import Path
import sys

_kernel = Path(__file__).resolve().parents[2] / '第三问-算法1'
if not (_kernel / 'q3/core.py').is_file():
    raise ImportError('Algorithm two requires the sibling 第三问-算法1/q3 kernel')
sys.path.insert(0, str(_kernel))
