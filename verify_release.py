from pathlib import Path
import hashlib,json

def main():
    root=Path(__file__).resolve().parent
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name,digest in manifest.items():
        if sha(root/name)!=digest:raise ValueError('Release mismatch: '+name)
    frozen=json.loads((root/'results/FULL_RUN_FREEZE.json').read_text(encoding='utf-8'))
    for name,digest in frozen['files'].items():
        if sha(root/name.replace('\\','/'))!=digest:raise ValueError('Frozen core mismatch: '+name)
    print('PASS:',len(manifest),'release files; all frozen core and model hashes match')

if __name__=='__main__':main()
