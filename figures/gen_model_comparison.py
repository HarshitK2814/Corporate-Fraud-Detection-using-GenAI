"""
Generate Model Performance Comparison Figure
Shows F1 Score and AUC-ROC grouped by model (not by metric)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data
models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'MLP Neural Network']
f1_scores = [85.2, 82.1, 90.5, 84.6]
auc_roc = [98.5, 98.4, 97.6, 98.2]

# Colors for each model (matching the original image)
colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#1f77b4']  # Green, Orange, Purple, Blue

# Setup figure
fig, ax = plt.subplots(figsize=(10, 6))

# Parameters
n_models = len(models)
bar_width = 0.35
group_spacing = 0.8  # Space between model groups
x = np.arange(n_models) * (2 * bar_width + group_spacing)

# Create bars
bars_f1 = ax.bar(x - bar_width/2, f1_scores, bar_width, label='F1 Score', color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
bars_auc = ax.bar(x + bar_width/2, auc_roc, bar_width, label='AUC-ROC', color=colors, edgecolor='black', linewidth=0.5, alpha=0.6, hatch='//')

# Add value labels on bars
def add_labels(bars, values):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

add_labels(bars_f1, f1_scores)
add_labels(bars_auc, auc_roc)

# Customize axes
ax.set_ylabel('Performance Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison Across Metrics', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0, 110)

# Add legend
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Add grid
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save figure
output_path = os.path.join(OUTPUT_DIR, 'model_comparison_grouped.png')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'[✓] Model comparison figure saved: {output_path}')
