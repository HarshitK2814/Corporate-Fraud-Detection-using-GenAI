"""Fig 9 (paper Fig 6) – SHAP Waterfall 2x2 — Light Theme"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.6})
BG='#FFFFFF'

features=['text_evasion','geo_shell','jitter_z','shimmer_z','pitch_var_z','pause_z']
companies={
    'Alphabet Inc\nCRDI = 1.94 (CREDIBLE)':
        {'base':0.181,'shap':[-0.25,-0.15,-0.05,-0.02,-0.01,-0.01],'pred':0.019},
    'JetBlue Airways\nCRDI = 35.2 (WATCHLIST)':
        {'base':0.181,'shap':[0.02,-0.05,0.12,0.04,0.01,0.03],'pred':0.352},
    'Wirecard AG\nCRDI = 81.4 (HIGH RISK)':
        {'base':0.181,'shap':[0.08,0.44,0.05,0.02,0.01,0.03],'pred':0.814},
    'Refco Inc\nCRDI = 78.6 (HIGH RISK)':
        {'base':0.181,'shap':[0.42,0.10,0.06,0.01,0.02,0.01],'pred':0.786},
}

fig,axes=plt.subplots(2,2,figsize=(7.08,5.5),facecolor=BG)
for ax,(name,data) in zip(axes.flat,companies.items()):
    ax.set_facecolor('#fafafa')
    shap_vals=data['shap']
    sorted_idx=np.argsort(np.abs(shap_vals))[::-1]
    cumulative=data['base']
    for rank,idx in enumerate(sorted_idx):
        val=shap_vals[idx]
        color='#E74C3C' if val>0 else '#27AE60'
        ax.barh(rank,val,left=cumulative,height=0.5,color=color,alpha=0.75,edgecolor=color,linewidth=0.8)
        sign='+' if val>0 else ''
        ax.text(cumulative+val/2,rank,f'{sign}{val:.2f}',ha='center',va='center',fontsize=6,fontweight='bold',color='#222')
        cumulative+=val
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in sorted_idx],fontsize=6.5)
    ax.axvline(data['base'],color='#aaa',ls='--',lw=0.7)
    ax.axvline(data['pred'],color='#D4AC0D',ls='-',lw=1.2)
    ax.text(data['base'],-0.7,f"E[f(x)]={data['base']:.3f}",ha='center',color='#888',fontsize=5.5)
    ax.text(data['pred'],len(features)-0.3,f"f(x)={data['pred']:.3f}",ha='center',color='#8B6914',fontsize=5.5)
    ax.set_title(name,fontsize=7.5,fontweight='bold',pad=4)
    ax.grid(True,axis='x',alpha=0.15); ax.invert_yaxis()

fig.suptitle('Fig. 6.  SHAP Waterfall Explanations',fontsize=10,fontweight='bold',y=0.99)
plt.tight_layout(rect=[0,0,1,0.95])
for ext in ['png','pdf']:
    fig.savefig(f'fig9_shap_waterfall.{ext}',dpi=300,bbox_inches='tight',pad_inches=0.1,facecolor=BG)
print("✓ fig9_shap_waterfall")
