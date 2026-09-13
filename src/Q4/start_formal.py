"""Q4 Ultra S21 probes; one user-started formal case, no automatic test creation."""
from pathlib import Path
import os
import shutil
import practice_windows as practice
from formal_session import main
from s21_layout import install

if os.name == 'nt':
    practice.POWERSHELL=shutil.which('powershell.exe') or str(Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe')

def build():
    install()
    policy=practice.make_policy('probes',practice.Config())
    return policy,practice.Belief(),practice.load('client').HttpClient

if __name__=='__main__':
    raise SystemExit(main(4,Path(__file__).resolve().parent,build,practice.inspect_ui,6000))
