"""Fig 7 – Cohen's d Effect Size Chart — Light Theme"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.6})
BG='#FFFFFF'

features=['text_evasion (ℓ)','geo_shell_risk (g)','audio_jitter_z (j)',
          'audio_pitch_var_z (v)','audio_shimmer_z (s)','audio_pause_z (p)']
cohens_d=[2.65,1.18,0.72,0.65,0.61,0.61]
mcols=['#D35400','#27AE60','#2980B9','#2980B9','#2980B9','#2980B9']

fig,ax=plt.subplots(figsize=(5.0,3.2),facecolor=BG)
ax.set_facecolor('#fafafa')
y=np.arange(len(features))
bars=ax.barh(y,cohens_d,height=0.55,color=mcols,alpha=0.8,edgecolor=mcols,linewidth=1.0)

for thresh,lbl in [(0.2,'Small'),(0.5,'Medium'),(0.8,'Large')]:
    ax.axvline(thresh,color='#bbb',ls='--',lw=0.7)
    ax.text(thresh,len(features)-0.3,lbl,ha='center',va='bottom',color='#888',fontsize=5.5)

for bar,d in zip(bars,cohens_d):
    sz='Large' if d>=0.8 else ('Medium' if d>=0.5 else 'Small')
    ax.text(bar.get_width()+0.05,bar.get_y()+bar.get_height()/2,
            f'd = {d:.2f} ({sz})',va='center',ha='left',fontsize=7,fontweight='bold',color='#222')

ax.set_yticks(y); ax.set_yticklabels(features,fontsize=8)
ax.set_xlabel("Cohen's d"); ax.set_xlim(0,3.2)
ax.set_title("Fig. 7.  Cohen's d Effect Sizes",fontsize=10,fontweight='bold')
ax.grid(True,axis='x',alpha=0.2); ax.invert_yaxis()
handles=[mp.Patch(color='#D35400',label='Linguistic'),mp.Patch(color='#27AE60',label='Geospatial'),
         mp.Patch(color='#2980B9',label='Audio')]
ax.legend(handles=handles,fontsize=6.5,loc='lower right',framealpha=0.9,edgecolor='#ccc')
plt.tight_layout()
for ext in ['png','pdf']:
    fig.savefig(f'fig7_cohens_d.{ext}',dpi=300,bbox_inches='tight',pad_inches=0.1,facecolor=BG)
print("✓ fig7_cohens_d")
