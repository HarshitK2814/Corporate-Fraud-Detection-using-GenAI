"""Fig 4 – CRDI Score Distribution (Violin) — Light Theme"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(42)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.6})
BG='#FFFFFF'

fraud_scores = np.clip(np.random.beta(5,2,50)*85+15, 5, 98)
legit_scores = np.clip(np.random.beta(2,6,226)*60+2, 0, 65)
fraud_scores[:20] = np.clip(np.random.beta(3,4,20)*50+10, 15, 55)

fig,ax=plt.subplots(figsize=(4.5,3.5),facecolor=BG)
ax.set_facecolor('#fafafa')

parts=ax.violinplot([legit_scores,fraud_scores],positions=[1,2],
                    showmeans=True,showmedians=True,widths=0.7)
for i,pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#27AE60','#C0392B'][i])
    pc.set_alpha(0.35); pc.set_edgecolor(['#1E8449','#922B21'][i]); pc.set_linewidth(1.2)
for pn in ['cmeans','cmedians','cmins','cmaxes','cbars']:
    if pn in parts: parts[pn].set_color('#555'); parts[pn].set_linewidth(0.8)

for i,(data,col) in enumerate([(legit_scores,'#27AE60'),(fraud_scores,'#C0392B')]):
    jitter=np.random.normal(0,0.06,len(data))
    ax.scatter(np.full_like(data,i+1)+jitter,data,c=col,alpha=0.35,s=8,edgecolors='none',zorder=3)

bands=[(0,30,'#27AE60',0.06),(30,50,'#F1C40F',0.06),(50,70,'#E67E22',0.06),
       (70,85,'#E74C3C',0.06),(85,100,'#922B21',0.06)]
for lo,hi,c,a in bands:
    ax.axhspan(lo,hi,color=c,alpha=a,zorder=0)
for y,lbl in [(15,'CREDIBLE'),(40,'WATCHLIST'),(60,'SUSPICIOUS'),(77,'HIGH RISK'),(92,'CRITICAL')]:
    ax.text(2.65,y,lbl,color='#666',fontsize=5.5,fontweight='bold',va='center')

ax.set_xticks([1,2]); ax.set_xticklabels(['Legitimate\n(n=226)','Fraud\n(n=50)'])
ax.set_ylabel('CRDI Score'); ax.set_ylim(-2,105)
ax.set_title('Fig. 4.  CRDI Score Distribution by Class',fontsize=10,fontweight='bold')
ax.grid(True,axis='y',alpha=0.2)

ax.annotate('20 "competent liars"\n(low CRDI despite fraud)',xy=(2,32),
            xytext=(2.45,20),fontsize=6,color='#8B6914',
            arrowprops=dict(arrowstyle='->',color='#8B6914',lw=0.8),
            bbox=dict(boxstyle='round,pad=0.3',fc='#FFF9E6',ec='#D4AC0D',lw=0.6))
plt.tight_layout()
for ext in ['png','pdf']:
    fig.savefig(f'fig4_crdi_violin.{ext}',dpi=300,bbox_inches='tight',pad_inches=0.1,facecolor=BG)
print("✓ fig4_crdi_violin")
