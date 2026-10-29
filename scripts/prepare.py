"""Freeze source-level splits BEFORE model selection; exclude prior test sources.

All existing blinkeeg epochs are historical development data. New public test
sources exclude both earlier public test lists, exact and sign/scale-equivalent
duplicates. Subject metadata unavailable: this is NOT subject-disjoint testing.
"""
from pathlib import Path
import sys,os,json,hashlib,argparse
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import numpy as np
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from awrls3.dictionary import ShapeDictionary


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    x=row-np.mean(row); x=x/(np.linalg.norm(x)+1e-30)
    if x[np.argmax(np.abs(x))]<0:x=-x
    return hashlib.sha256(np.round(x,10).tobytes()).hexdigest()


def split_sources(data,excluded,rng):
    keys=[key(x) for x in data];blocked={keys[i] for i in excluded}
    groups={}
    for i,k in enumerate(keys):
        if k not in blocked: groups.setdefault(k,[]).append(i)
    # Use one representative per equivalence group, so no group crosses splits.
    reps=np.array([v[0] for v in groups.values()]);rng.shuffle(reps)
    n=len(reps);a=int(.65*n);b=int(.8*n)
    return {'train':reps[:a].tolist(),'validation':reps[a:b].tolist(),
            'test':reps[b:].tolist(),'excluded_prior':sorted(excluded),
            'unique_available':n,'duplicate_equivalence':'center, L2 norm, sign, round 10 decimals',
            'excluded_equivalent_count':sum(k in blocked for k in keys)}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,default=ROOT.parent/'awrls2')
    ap.add_argument('--internal',type=Path,default=Path('C:/Users/13665/Desktop/eegqz/databasic/blinkeeg'))
    args=ap.parse_args();previous=args.previous
    data=previous/'benchmark_data';rng=np.random.default_rng(20261028)
    arrays={k:np.load(data/f'{k.upper()}_all_epochs.npy') for k in ['eeg','eog']}
    exclusions={k:set() for k in arrays}
    for dirname in ['eegdenoisenet','eegdenoisenet_official_mixing']:
        p=json.loads((previous/'awrls2_package/results'/dirname/'protocol.json').read_text())
        for k in arrays:exclusions[k].update(p[k+'_idx'])
    splits={k:split_sources(v,exclusions[k],rng) for k,v in arrays.items()}
    order=rng.permutation(300)
    splits['internal']={'train':order[:150].tolist(),'validation':order[150:].tolist(),
                        'status':'All 300 previously analysed; retrospective development only.'}
    splits['seed']=20261028
    splits['paths']={'public':str(data.resolve()),'internal':str(args.internal.resolve())}
    splits['sha256']={str(data/f'{k.upper()}_all_epochs.npy'):sha(data/f'{k.upper()}_all_epochs.npy') for k in arrays}
    splits['limitations']=['No subject mapping; source-disjoint not subject-disjoint.',
                           'Does not detect all approximate/overlapping source epochs.',
                           'Historical internal data used for training and validation, never called fresh test.']
    (ROOT/'results/source_split.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
    train_eog=arrays['eog'][splits['eog']['train']]
    s=np.load(args.internal/'clean.npy');x=np.load(args.internal/'mixed.npy')
    internal_art=resample_poly((x-s)[splits['internal']['train']],32,25,axis=1)
    dictionary=ShapeDictionary.train(list(train_eog)+list(internal_art),256,16,
        {'split_sha256':sha(ROOT/'results/source_split.json'),'source':'training EOG and historical internal training artifact x-s',
         'eog_indices':splits['eog']['train'],'internal_indices':splits['internal']['train']})
    dictionary.save(ROOT/'models/dictionary.npz')
    print({k:{s:len(v[s]) for s in ['train','validation','test']} for k,v in splits.items() if k in ['eeg','eog']})
    print('Dictionary',dictionary.shapes.shape,flush=True)
