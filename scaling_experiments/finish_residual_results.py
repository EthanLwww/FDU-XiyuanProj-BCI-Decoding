"""Run once after the suite completes: compare, preserve analysis sources, archive."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from compare_scaling_models import ROOT, BASELINE, compare
from streaming_data import save_json

def main():
    assert json.loads((ROOT/'suite.status.json').read_text())['status']=='complete'
    analysis=ROOT/'analysis_source';analysis.mkdir(exist_ok=True)
    for name in ['compare_scaling_models.py','finish_residual_results.py','monitor_residual.py']:
        shutil.copy2(Path(__file__).parent/name,analysis/name)
    save_json(ROOT/'analysis_source_sha256.json',
              {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in analysis.glob('*.py')})
    compare(ROOT,BASELINE)
    archive=ROOT/'comparison_bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(ROOT.rglob('*')):
            rel=p.relative_to(ROOT)
            if not p.is_file() or rel.parts[0] in ('cache','runtime') or '__pycache__' in rel.parts:
                continue
            if p.name.endswith('.tar.gz') or p.name=='comparison_bundle.sha256':
                continue
            tar.add(p,arcname=str(rel))
        for p in sorted((ROOT/'cache').glob('S*/*.json')):
            if p.name in ('ready.json','baseline_verified.json'):
                tar.add(p,arcname=str(Path('data_manifests')/p.parent.name/p.name))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (ROOT/'comparison_bundle.sha256').write_text(digest+'  comparison_bundle.tar.gz\n')
    print(json.dumps(dict(bundle=str(archive),sha256=digest,bytes=archive.stat().st_size)))

if __name__=='__main__':
    main()
