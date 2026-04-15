"""Fig 8 – Confusion Matrix — Light Theme"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.6})
BG='#FFFFFF'

cm_full=np.array([[224,2],[7,43]])
cm_text=np.array([[210,16],[6,44]])
cmap=LinearSegmentedColormap.from_list('lp',['#F5EEF8','#D2B4DE','#8E44AD'])

fig,axes=plt.subplots(1,2,figsize=(7.08,3.2),facecolor=BG)
for idx,(matrix,title) in enumerate([(cm_full,'(a) Full CRDI Model'),(cm_text,'(b) Text-Only Baseline')]):
    ax=axes[idx]; ax.set_facecolor('#fafafa')
    im=ax.imshow(matrix,cmap=cmap,aspect='auto')
    for i in range(2):
        for j in range(2):
            v=matrix[i,j]; pct=v/matrix.sum()*100
            ax.text(j,i,f'{v}\n({pct:.1f}%)',ha='center',va='center',fontsize=11,fontweight='bold',
                    color='white' if v>100 else '#222')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred: Legit','Pred: Fraud'],fontsize=8)
    ax.set_yticklabels(['True: Legit','True: Fraud'],fontsize=8)
    ax.set_title(title,fontsize=9,fontweight='bold',pad=8)
    tp,fp,fn,tn=matrix[1,1],matrix[0,1],matrix[1,0],matrix[0,0]
    p=tp/(tp+fp); r=tp/(tp+fn); f1=2*p*r/(p+r)
    ax.text(0.5,-0.22,f'Prec={p:.3f}  Rec={r:.3f}  F1={f1:.3f}',
            ha='center',va='top',transform=ax.transAxes,fontsize=7,color='#555')

fig.suptitle('Fig. 8.  Confusion Matrix Comparison',fontsize=10,fontweight='bold',y=0.98)
plt.tight_layout(rect=[0,0.02,1,0.93])
for ext in ['png','pdf']:
    fig.savefig(f'fig8_confusion_matrix.{ext}',dpi=300,bbox_inches='tight',pad_inches=0.1,facecolor=BG)
print("✓ fig8_confusion_matrix")
