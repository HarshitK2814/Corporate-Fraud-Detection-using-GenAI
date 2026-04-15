"""Fig 6 – Live Pipeline CRDI Score Bar Chart — Light Theme"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.6})
BG='#FFFFFF'

companies=['Alphabet Inc','JPMorgan Chase','Proto Labs','Abercrombie & Fitch',
           'JetBlue Airways','Chevron','Editas Medicine','Zillow Group']
scores=[1.94,8.5,12.3,22.7,35.2,42.1,62.5,95.12]
dominants=['Geo ✓ Audio ✓ Text ✓','Geo ✓ Audio ✓ Text ✓',
           'Geo ✓ Audio ✓ Text ✓','Text evasion ↑',
           'Audio stress ↑','Text defensive ↑ Audio ↑',
           'Text evasive ↑↑ Audio ↑↑','Geo anomaly ↑↑ Audio ↑↑']

def band(s):
    if s<30: return '#27AE60','CREDIBLE'
    elif s<50: return '#D4AC0D','WATCHLIST'
    elif s<70: return '#E67E22','SUSPICIOUS'
    elif s<85: return '#E74C3C','HIGH RISK'
    else: return '#922B21','CRITICAL'

fig,ax=plt.subplots(figsize=(7.08,4.0),facecolor=BG)
ax.set_facecolor('#fafafa')
y=np.arange(len(companies))

for i in range(len(companies)):
    c,lbl=band(scores[i])
    ax.barh(i,scores[i],height=0.6,color=c,alpha=0.8,edgecolor=c,linewidth=1.0)
    ax.text(scores[i]+1.5,i,f'{scores[i]:.1f}  ({lbl})',va='center',ha='left',fontsize=7,fontweight='bold',color='#222')

for lo,hi,c in [(0,30,'#27AE60'),(30,50,'#D4AC0D'),(50,70,'#E67E22'),(70,85,'#E74C3C'),(85,100,'#922B21')]:
    ax.axvspan(lo,hi,color=c,alpha=0.05,zorder=0)

ax.set_yticks(y); ax.set_yticklabels(companies,fontsize=8)
ax.set_xlabel('CRDI Score (0–100)'); ax.set_xlim(0,120)
ax.set_title('Fig. 6.  Live Pipeline CRDI Scores — 8 Real Companies',fontsize=10,fontweight='bold')
ax.grid(True,axis='x',alpha=0.2); ax.invert_yaxis()
plt.tight_layout()
for ext in ['png','pdf']:
    fig.savefig(f'fig6_crdi_bar.{ext}',dpi=300,bbox_inches='tight',pad_inches=0.1,facecolor=BG)
print("✓ fig6_crdi_bar")
