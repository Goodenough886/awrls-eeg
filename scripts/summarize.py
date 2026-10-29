"""Paired source-level statistics; never count SNR repeats as independent."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scipy.stats import wilcoxon


def paired(df,unit,group):
    rows=[];rng=np.random.default_rng(20261028)
    for metric in ['rrmse_t','rrmse_s','cc','dbp_total']:
        wide=df.groupby([unit,'variant'])[metric].mean().unstack('variant')
        for candidate in [c for c in wide if c!='V2' and c!='INPUT']:
            for reference in ['V2','TCN_CONTROL']:
                if reference not in wide or candidate==reference:continue
                pair=wide[[reference,candidate]].dropna();delta=(pair[candidate]-pair[reference]).to_numpy()
                if len(delta)<2:continue
                n=len(delta);boot=np.mean(delta[rng.integers(0,n,size=(5000,n))],axis=1)
                p=1. if np.all(abs(delta)<1e-15) else float(wilcoxon(delta,zero_method='pratt').pvalue)
                rows.append(dict(group=group,metric=metric,candidate=candidate,reference=reference,n=n,
                    mean_reference=pair[reference].mean(),mean_candidate=pair[candidate].mean(),
                    delta_candidate_minus_reference=delta.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),p=p))
    out=pd.DataFrame(rows)
    # Holm within each explicitly identified group/reference comparison family.
    for ref,ix in out.groupby('reference').groups.items():
        order=out.loc[ix,'p'].sort_values().index;adjust=np.maximum.accumulate(out.loc[order,'p'].to_numpy()*np.arange(len(order),0,-1))
        out.loc[order,'p_holm']=np.minimum(adjust,1)
    return out


if __name__=='__main__':
    results=ROOT/'results';all_stats=[]
    for split in ['validation','test']:
        df=pd.read_csv(results/f'{split}_models.csv')
        # Primary seed retained; repeats are explicit sensitivity comparisons.
        mask=df.condition.str.startswith('public_snr_')
        all_stats.append(paired(df[mask],'epoch',split+'_public_snr_averaged'))
        all_stats.append(paired(df[df.condition=='public_clean'],'epoch',split+'_public_clean'))
        if split=='validation':
            for cond in ['internal_native','internal_clean']:
                all_stats.append(paired(df[df.condition==cond],'epoch',cond+'_development'))
        summary=df.groupby(['condition','variant'])[['rmse','rrmse_t','rrmse_s','cc','dbp_total','changed','changed_gt_5pct_input_rms']].mean()
        summary.to_csv(results/f'{split}_model_summary.csv')
        primary=df[mask].groupby('variant')[['rrmse_t','rrmse_s','cc','dbp_total']].mean()
        primary.to_csv(results/f'{split}_public_snr_averaged.csv')
        print(split,primary.to_string())
        repeat=df[mask & df.variant.str.startswith(('V3_HYBRID','TCN_CONTROL'))].copy()
        repeat['family']=repeat.variant.str.replace(r'_seed\d+$','',regex=True)
        repeat['seed']=repeat.variant.str.extract(r'_seed(\d+)$')[0].fillna('20261028')
        seed_means=repeat.groupby(['family','seed'])[['rrmse_t','rrmse_s','cc','dbp_total']].mean()
        seed_means.to_csv(results/f'{split}_seed_means.csv')
        seed_means.groupby('family').agg(['mean','std','min','max']).to_csv(results/f'{split}_seed_variation.csv')
    real=pd.read_csv(results/'kaya_30s_by_subject.csv')
    all_stats.append(paired(real,'subject','kaya_30s_silver_subject_averaged'))
    pd.concat(all_stats,ignore_index=True).to_csv(results/'paired_statistics.csv',index=False)
