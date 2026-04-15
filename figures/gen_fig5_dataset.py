"""Fig 5 – Dataset Composition — Light Theme"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.6})
BG='#FFFFFF'

fig,ax=plt.subplots(figsize=(6.5,3.8),facecolor=BG)
ax.set_facecolor('#fafafa')

categories=['Total Dataset\n(N=276)','Legitimate\n(n=226)','Fraud\n(n=50)',
            'Standard Fraud\n(n=30)','Competent Liars\n(n=20)']
y_pos=[4,3,2,1,0]
widths=[276,226,50,30,20]
colors_bar=['#3498DB','#27AE60','#E74C3C','#C0392B','#F39C12']

for i in range(len(y_pos)):
    ax.barh(y_pos[i],widths[i],height=0.55,color=colors_bar[i],alpha=0.8,
            edgecolor=colors_bar[i],linewidth=1.2)
    ax.text(widths[i]+3,y_pos[i],f'n={widths[i]}',va='center',ha='left',fontsize=8,fontweight='bold')

ax.set_yticks(y_pos); ax.set_yticklabels(categories,fontsize=8)
ax.set_xlabel('Number of Company Profiles')
ax.set_title('Fig. 5.  Dataset Composition',fontsize=10,fontweight='bold')
ax.set_xlim(0,310)

tier_x=0
for w,c,lbl in [(152,'#1ABC9C','Enterprise (152)'),(101,'#F39C12','SME (101)'),(23,'#E74C3C','Startup (23)')]:
    ax.barh(4,w,left=tier_x,height=0.25,color=c,alpha=0.5,edgecolor=c)
    if w>30: ax.text(tier_x+w/2,4+0.18,lbl,ha='center',va='bottom',fontsize=5.5,color='#333')
    tier_x+=w

handles=[mpatches.Patch(color='#1ABC9C',alpha=0.5,label='Enterprise (152)'),
         mpatches.Patch(color='#F39C12',alpha=0.5,label='SME (101)'),
         mpatches.Patch(color='#E74C3C',alpha=0.5,label='Startup (23)')]
ax.legend(handles=handles,fontsize=6,loc='lower right',title='Size Tiers',
          title_fontsize=6.5,framealpha=0.9,edgecolor='#ccc')
ax.grid(True,axis='x',alpha=0.2)

ax.annotate('Fraud prevalence: 18.1%\n(class imbalance ≈ 4.5:1)',
            xy=(50,2.3),xytext=(150,1.5),fontsize=6.5,color='#8B6914',
            arrowprops=dict(arrowstyle='->',color='#8B6914',lw=0.8),
            bbox=dict(boxstyle='round,pad=0.3',fc='#FFF9E6',ec='#D4AC0D',lw=0.6))
plt.tight_layout()
for ext in ['png','pdf']:
    fig.savefig(f'fig5_dataset_composition.{ext}',dpi=300,bbox_inches='tight',pad_inches=0.1,facecolor=BG)
print("✓ fig5_dataset_composition")
