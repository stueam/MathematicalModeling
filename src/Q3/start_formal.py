"""Q3 bayes-fast; explicit connection to one user-started formal case only."""

from pathlib import Path
import practice_windows as practice
from formal_session import main


def build():
    policy = practice.Policy(practice.Config())
    return policy, practice.CoupledBelief(), practice.load('client').HttpClient


if __name__ == '__main__':
    raise SystemExit(main(3, Path(__file__).resolve().parent, build, practice.inspect_ui, 4000))
