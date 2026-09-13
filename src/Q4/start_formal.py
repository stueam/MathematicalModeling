"""Q4 Ultra S21 probes; one user-started formal case, no automatic test creation."""

from pathlib import Path
import practice_windows as practice
from formal_session import main


def build():
    policy = practice.Policy(practice.Config())
    return policy, practice.Belief(), practice.load('client').HttpClient


if __name__ == '__main__':
    raise SystemExit(main(4, Path(__file__).resolve().parent, build, practice.inspect_ui, 6000))
