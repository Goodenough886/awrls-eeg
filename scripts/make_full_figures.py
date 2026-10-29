from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/figures'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
NAMES=['V2','V3_full','V3_HYBRID','TCN_CONTROL'];LABELS=['AWRLS-2','AWRLS-3 classical','Hybrid TCN','Standalone TCN']
COLORS=['#777777','#009E73','#0072B2','#D55E00']

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    real=pd.read_csv(ROOT/'results/full_kaya/by_subject.csv');real=real[real.kind=='silver']
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,metric,label in zip(axs,['silver_rrmse_t','silver_cc','silver_dbp_total'],['Temporal RRMSE','Correlation','Band-power deviation']):
        wide=real.pivot(index='subject',columns='variant',values=metric)
        for _,row in wide.iterrows():ax.plot(range(4),row[NAMES],color='#bbbbbb',linewidth=.7,alpha=.65)
        for j,(name,color) in enumerate(zip(NAMES,COLORS)):ax.scatter(np.full(len(wide),j),wide[name],color=color,s=20)
        ax.set_xticks(range(4),LABELS,rotation=25,ha='right');ax.set_ylabel(label)
    fig.suptitle('Full Kaya records: 13 subjects; ICA silver-reference agreement, not ground truth')
    save(fig,'05_kaya_full_silver')
    df=pd.read_csv(ROOT/'results/full_internal/by_epoch.csv')
    df=df[(df.split_role=='historical_development')&df.condition.astype(str).isin(['-7','-5','-3','-1','1','3'])].copy()
    df['snr']=df.condition.astype(int)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name,color,label in zip(NAMES,COLORS,LABELS):
        g=df[df.variant==name].groupby('snr')[['rmse','cc']].mean()
        for ax,metric in zip(axs,['rmse','cc']):ax.plot(g.index,g[metric],'o-',color=color,label=label,markersize=4)
    axs[0].set_ylabel('RMSE');axs[1].set_ylabel('Correlation')
    for ax in axs:ax.set_xlabel('Physical power SNR (dB)');ax.set_xticks([-7,-5,-3,-1,1,3])
    axs[0].legend(frameon=False,fontsize=8);fig.suptitle('Historical internal development: 150 epochs; not an independent test')
    save(fig,'06_internal_full_snr')
    p=pd.read_csv(ROOT/'results/full_public_all/summary_by_role.csv')
    p=p[p.condition.str.startswith('public_snr_')]
    g=p.groupby(['split_role','variant']).rrmse_t.mean()
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    roles=['train','validation','test','historical_prior'];x=np.arange(4)
    for j,(name,color,label) in enumerate(zip(NAMES,COLORS,LABELS)):
        ax.bar(x+(j-1.5)*.18,[g.loc[(r,name)] for r in roles],width=.18,color=color,label=label)
    ax.set_xticks(x,['Training sources','Validation sources','All test-role sources*','Historically viewed']);ax.set_ylabel('Temporal RRMSE, SNR-averaged')
    ax.set_ylim(0,1.30);ax.legend(frameon=False,fontsize=8,loc='upper center',ncol=4);ax.set_title('All 4,514 EEG and 3,400 EOG sources: descriptive coverage by role')
    fig.text(.5,-.05,'*Primary independent source-pair result remains the original 554 pairs; additional pairs can reuse EOG.',ha='center',fontsize=8)
    save(fig,'07_public_all_source_coverage')
