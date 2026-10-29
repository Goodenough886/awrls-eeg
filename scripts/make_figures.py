"""Source-level means, 95% bootstrap CI, and explicit training-seed sensitivity."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
COLORS={'V2':'#777777','V3_full':'#009E73','V3_HYBRID':'#0072B2','TCN_CONTROL':'#D55E00'}
LABELS={'V2':'AWRLS-2','V3_full':'AWRLS-3 classical','V3_HYBRID':'Hybrid TCN','TCN_CONTROL':'Standalone TCN'}

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    test=pd.read_csv(ROOT/'results/test_models.csv');rng=np.random.default_rng(20261028)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for name,color in COLORS.items():
        df=test[(test.variant==name)&test.condition.str.startswith('public_snr_')].copy()
        df['snr']=df.condition.str.rsplit('_',n=1).str[-1].astype(int)
        for ax,metric in zip(axs,['rrmse_t','cc']):
            means=[];lo=[];hi=[]
            for snr,g in df.groupby('snr'):
                a=g[metric].to_numpy();boot=a[rng.integers(0,len(a),(2000,len(a)))].mean(1)
                means.append(a.mean());lo.append(np.quantile(boot,.025));hi.append(np.quantile(boot,.975))
            xs=sorted(df.snr.unique());ax.plot(xs,means,'o-',color=color,label=LABELS[name],markersize=4)
            ax.fill_between(xs,lo,hi,color=color,alpha=.1);ax.set_xlabel('Physical power SNR (dB)');ax.set_xticks(xs)
    axs[0].set_ylabel('Temporal RRMSE (lower is better)');axs[1].set_ylabel('Correlation (higher is better)')
    axs[0].legend(fontsize=8,frameon=False);fig.suptitle('New source-disjoint test: 554 EEG/EOG pairs; primary seed')
    save(fig,'01_public_test')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    val=pd.read_csv(ROOT/'results/validation_models.csv')
    for ax,data,cond,title in [(axs[0],val,'internal_clean','Historical internal validation'),(axs[1],test,'public_clean','New public test')]:
        for j,(name,color) in enumerate(COLORS.items()):
            a=data[(data.variant==name)&(data.condition==cond)].rrmse_t.to_numpy()
            boot=a[rng.integers(0,len(a),(2000,len(a)))].mean(1)
            ax.bar(j,a.mean(),color=color,width=.65)
            ax.errorbar(j,a.mean(),yerr=[[a.mean()-np.quantile(boot,.025)],[np.quantile(boot,.975)-a.mean()]],color='black',capsize=3)
        ax.set_xticks(range(4),[LABELS[k] for k in COLORS],rotation=20,ha='right');ax.set_title(title);ax.set_ylabel('Clean-input temporal RRMSE')
    save(fig,'02_clean_preservation')
    seeds=pd.read_csv(ROOT/'results/test_seed_means.csv')
    fig,ax=plt.subplots(figsize=(5.5,3.8),layout='constrained')
    for j,name in enumerate(['V3_HYBRID','TCN_CONTROL']):
        a=seeds[seeds.family==name]
        ax.scatter(np.full(len(a),j),a.rrmse_t,color=COLORS[name],s=45)
        for _,r in a.iterrows():ax.annotate(str(int(r.seed)),(j,r.rrmse_t),xytext=(8,0),textcoords='offset points',fontsize=8)
    ax.set_xlim(-.4,1.7);ax.set_xticks([0,1],['Hybrid TCN','Standalone TCN']);ax.set_ylabel('Test temporal RRMSE, averaged across SNRs')
    ax.set_title('Three training seeds; same frozen test sources');save(fig,'03_seed_sensitivity')
    real=pd.read_csv(ROOT/'results/kaya_30s_by_subject.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,metric in zip(axs,['rrmse_t','cc']):
        table=real.pivot(index='subject',columns='variant',values=metric)
        names=list(COLORS)
        for _,row in table.iterrows():ax.plot(range(4),row[names],color='#bbbbbb',alpha=.6,linewidth=.8)
        for j,name in enumerate(names):ax.scatter(np.full(len(table),j),table[name],color=COLORS[name],s=20,label=LABELS[name])
        ax.set_xticks(range(4),[LABELS[n] for n in names],rotation=20,ha='right');ax.set_ylabel('Silver-reference '+metric)
    fig.suptitle('Kaya: 13 subjects, first 30 seconds per record; ICA is not ground truth')
    save(fig,'04_kaya_silver')
