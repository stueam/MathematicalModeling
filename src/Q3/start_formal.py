"""Q3 bayes-fast; explicit connection to one user-started formal case only."""
from pathlib import Path
import os
import shutil
import practice_windows as practice
from formal_session import main

if os.name == 'nt':
    practice.POWERSHELL=shutil.which('powershell.exe') or str(Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe')

def build():
    policy=practice.make_policy('bayes-fast',practice.SectorConfig())
    return policy,practice.make_belief(policy),practice.load('client').HttpClient

if __name__=='__main__':
    raise SystemExit(main(3,Path(__file__).resolve().parent,build,practice.inspect_ui,4000))
