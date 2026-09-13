"""S21 Q4 practice launcher. --self-test is fully offline; --connect starts practice."""
import sys, os, shutil, json
from pathlib import Path
from s21_layout import install

def main():
    points=install()
    from q4.sector import ProbePolicy
    if '--self-test' in sys.argv:
        from q4.core import Belief
        p=ProbePolicy()
        action=p.choose(Belief())
        assert len(p.points)==21
        print(json.dumps({'coverage':'passed','stations':21,'implementation':p.implementation,
                          'first_action':action.payload(),'network_used':False},ensure_ascii=False))
        return
    if '--local' in sys.argv:
        import run
        sys.argv=[sys.argv[0],'local','--policy','probes']+[a for a in sys.argv[1:] if a!='--local']
        run.main()
        return
    import practice_windows as adapter
    if os.name=='nt':
        adapter.POWERSHELL=shutil.which('powershell.exe') or str(Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe')
    original_manifest=adapter.code_manifest
    def code_manifest():
        import hashlib
        result=original_manifest()
        for name in ('start_s21.py','s21_layout.py','s21_certificate.json','check_s21_certificate.py'):
            result[name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        return result
    adapter.code_manifest=code_manifest
    if '--policy' in sys.argv:
        raise SystemExit('This launcher fixes the S21 probes policy; omit --policy.')
    sys.argv += ['--policy','probes']
    adapter.main()

if __name__=='__main__':main()
