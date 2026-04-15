"""
Generate Publication-Quality Figures for Veritas AI Paper
=========================================================
Figures 1, 2, 4: Draw.io-style clean flowcharts (matplotlib)
Figures 3, 5: Data plots (matplotlib)

All outputs: 300 DPI, IEEE column width (7.16 in)

Usage:
    python scripts/generate_paper_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

# ── Global Style ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Draw.io inspired palette
C_NAVY      = '#1a365d'
C_DARK_BLUE = '#2b6cb0'
C_BLUE      = '#3182ce'
C_LIGHT_BLUE= '#bee3f8'
C_TEAL      = '#2c7a7b'
C_TEAL_LIGHT= '#b2f5ea'
C_PURPLE    = '#553c9a'
C_PURP_LIGHT= '#d6bcfa'
C_GREEN     = '#276749'
C_GREEN_LT  = '#c6f6d5'
C_ORANGE    = '#c05621'
C_ORANGE_LT = '#feebc8'
C_RED       = '#c53030'
C_RED_LT    = '#fed7d7'
C_CORAL     = '#e53e3e'
C_GRAY      = '#4a5568'
C_GRAY_LT   = '#edf2f7'
C_YELLOW_LT = '#fefcbf'
C_WHITE     = '#ffffff'
C_SLATE     = '#718096'

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Drawing Helpers (draw.io style) ──────────────────────────

def draw_box(ax, x, y, w, h, label, sublabel=None,
             fill=C_LIGHT_BLUE, border=C_DARK_BLUE, text_color=C_NAVY,
             fontsize=8, sublabel_fs=6.5, lw=1.5, radius=0.04, zorder=2):
    """Draw a draw.io-style rounded rectangle with label + optional sublabel."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=border, linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h*0.6, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=zorder+1)
        ax.text(x + w/2, y + h*0.30, sublabel, ha='center', va='center',
                fontsize=sublabel_fs, color=C_SLATE, zorder=zorder+1)
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=zorder+1)


def draw_arrow_v(ax, x, y1, y2, color=C_GRAY, lw=1.3):
    """Vertical arrow (downward)."""
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)


def draw_arrow_h(ax, x1, x2, y, color=C_GRAY, lw=1.3):
    """Horizontal arrow (right)."""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=1)


def draw_arrow_diag(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.2):
    """Diagonal arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0.08'), zorder=1)


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: System Architecture
# ═══════════════════════════════════════════════════════════════
def generate_fig1():
    fig, ax = plt.subplots(figsize=(7.16, 5.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 9)
    ax.axis('off')
    ax.set_aspect('equal')

    # ── Row 1: Input ──
    draw_box(ax, 3.0, 7.5, 4.0, 1.0,
             'Company Profile',
             'Name  ·  GPS  ·  Audio  ·  Transcript',
             fill=C_LIGHT_BLUE, border=C_NAVY, text_color=C_NAVY, fontsize=10)

    # ── Arrows down to 3 modules ──
    draw_arrow_diag(ax, 4.0, 7.5, 1.4, 6.6, color=C_TEAL)
    draw_arrow_v(ax, 5.0, 7.5, 6.6, color=C_DARK_BLUE)
    draw_arrow_diag(ax, 6.0, 7.5, 8.6, 6.6, color=C_PURPLE)

    # ── Row 2: Three Modules ──
    # Geospatial
    draw_box(ax, 0.0, 4.8, 3.0, 1.8,
             'Geospatial Module', None,
             fill=C_TEAL_LIGHT, border=C_TEAL, text_color=C_TEAL, fontsize=9)
    items_geo = ['Google Places API', 'Street View (4 angles)', 'ResNet-18 / Places365', 'Entity Name Match']
    for i, item in enumerate(items_geo):
        ax.text(1.5, 6.12 - i*0.28, item, ha='center', va='center',
                fontsize=6, color=C_GRAY, zorder=3)

    # Audio
    draw_box(ax, 3.5, 4.8, 3.0, 1.8,
             'Audio DSP Module', None,
             fill=C_LIGHT_BLUE, border=C_DARK_BLUE, text_color=C_DARK_BLUE, fontsize=9)
    items_audio = ['Librosa (YIN + RMS)', 'Jitter  ·  Shimmer', 'Pitch Variance', 'Pause Rate']
    for i, item in enumerate(items_audio):
        ax.text(5.0, 6.12 - i*0.28, item, ha='center', va='center',
                fontsize=6, color=C_GRAY, zorder=3)

    # Text
    draw_box(ax, 7.0, 4.8, 3.0, 1.8,
             'Linguistic Module', None,
             fill=C_PURP_LIGHT, border=C_PURPLE, text_color=C_PURPLE, fontsize=9)
    items_text = ['Llama 3.3 70B (Groq)', 'Chain-of-Thought Prompt', 'Semantic Evasion', 'Deception Score']
    for i, item in enumerate(items_text):
        ax.text(8.5, 6.12 - i*0.28, item, ha='center', va='center',
                fontsize=6, color=C_GRAY, zorder=3)

    # Feature outputs
    ax.text(1.5, 4.55, 'geo_shell_risk', ha='center', fontsize=6, color=C_TEAL,
            fontstyle='italic', fontfamily='monospace', zorder=3)
    ax.text(5.0, 4.55, '4 audio features', ha='center', fontsize=6, color=C_DARK_BLUE,
            fontstyle='italic', fontfamily='monospace', zorder=3)
    ax.text(8.5, 4.55, 'text_semantic_evasion', ha='center', fontsize=6, color=C_PURPLE,
            fontstyle='italic', fontfamily='monospace', zorder=3)

    # ── Arrows to fusion ──
    draw_arrow_diag(ax, 1.5, 4.8, 4.3, 4.0, color=C_TEAL)
    draw_arrow_v(ax, 5.0, 4.8, 4.0, color=C_DARK_BLUE)
    draw_arrow_diag(ax, 8.5, 4.8, 5.7, 4.0, color=C_PURPLE)

    # ── Row 3: Fusion Engine ──
    draw_box(ax, 2.5, 2.9, 5.0, 1.1,
             'ML Fusion Engine',
             'GradientBoosting  ·  6 features  ·  F1 = 0.905',
             fill=C_TEAL_LIGHT, border=C_TEAL, text_color=C_NAVY, fontsize=10)

    # ── Arrow to XAI ──
    draw_arrow_v(ax, 5.0, 2.9, 2.3, color=C_PURPLE)

    # ── Row 4: Explainability ──
    draw_box(ax, 2.0, 1.3, 6.0, 1.0,
             'Explainability Layer',
             'SHAP TreeExplainer  +  LLM Weight Rationale  +  LLM AI Summary',
             fill=C_PURP_LIGHT, border=C_PURPLE, text_color=C_PURPLE, fontsize=9)

    # ── Arrow to output ──
    draw_arrow_v(ax, 5.0, 1.3, 0.7, color=C_CORAL)

    # ── Row 5: CRDI Output ──
    draw_box(ax, 3.0, -0.15, 4.0, 0.85,
             'CRDI Score (0–100)',
             'CREDIBLE  ·  WATCHLIST  ·  SUSPICIOUS  ·  HIGH_RISK  ·  CRITICAL',
             fill=C_RED_LT, border=C_RED, text_color=C_RED, fontsize=10, sublabel_fs=5.5)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_system_architecture.png'))
    plt.close()
    print('  [✓] Figure 1: System Architecture')


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Geospatial Module Pipeline
# ═══════════════════════════════════════════════════════════════
def generate_fig2():
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    ax.set_xlim(-0.3, 11.3)
    ax.set_ylim(-0.5, 4.0)
    ax.axis('off')

    # ── Step 1: GPS Input ──
    draw_box(ax, 0.0, 2.0, 2.2, 1.4,
             'GPS Input', '(latitude, longitude)\nCompany Name',
             fill=C_LIGHT_BLUE, border=C_NAVY, text_color=C_NAVY, fontsize=9)

    draw_arrow_h(ax, 2.2, 2.8, 2.7, color=C_GRAY)

    # ── Step 2: Google APIs ──
    draw_box(ax, 2.8, 2.0, 2.2, 1.4,
             'Google APIs', 'Places API\nStatic Street View\n(0°, 90°, 180°, 270°)',
             fill=C_TEAL_LIGHT, border=C_TEAL, text_color=C_TEAL, fontsize=9, sublabel_fs=5.5)

    draw_arrow_h(ax, 5.0, 5.6, 2.7, color=C_GRAY)

    # ── Step 3: CNN Classifier ──
    draw_box(ax, 5.6, 2.0, 2.5, 1.4,
             'Places365 CNN', 'ResNet-18 (pretrained)\n365 scene categories',
             fill=C_ORANGE_LT, border=C_ORANGE, text_color=C_ORANGE, fontsize=9, sublabel_fs=5.5)

    draw_arrow_h(ax, 8.1, 8.7, 2.7, color=C_GRAY)

    # ── Step 4: Output ──
    draw_box(ax, 8.7, 2.0, 2.3, 1.4,
             'Shell Risk Score', '0.0 (Corporate) →\n1.0 (Suspicious)',
             fill=C_RED_LT, border=C_RED, text_color=C_RED, fontsize=9, sublabel_fs=6)

    # ── Risk mapping table (top) ──
    table_x = 5.6
    table_y = 3.55
    mappings = [
        ('office_building', '0.05', C_GREEN),
        ('skyscraper',      '0.03', C_GREEN),
        ('residential',     '0.70', C_ORANGE),
        ('vacant_lot',      '0.90', C_RED),
    ]
    ax.text(table_x + 1.25, table_y + 0.2, 'Risk Matrix', ha='center',
            fontsize=6.5, fontweight='bold', color=C_GRAY)
    for i, (scene, risk, color) in enumerate(mappings):
        x_pos = table_x + (i % 2) * 1.3
        y_pos = table_y - (i // 2) * 0.22
        ax.text(x_pos, y_pos, f'{scene}: ', ha='left', fontsize=5, color=C_GRAY,
                fontfamily='monospace', zorder=3)
        ax.text(x_pos + 1.05, y_pos, risk, ha='left', fontsize=5.5, color=color,
                fontweight='bold', fontfamily='monospace', zorder=3)

    # ── Entity Verification (bottom row) ──
    draw_box(ax, 1.0, -0.1, 9.0, 1.1,
             'Entity Name Verification', None,
             fill=C_YELLOW_LT, border='#b7791f', text_color='#744210', fontsize=8)

    ax.text(5.5, 0.5, 'Input: "GlobalTech LLC"    vs    Google Places: "Joe\'s Bakery"',
            ha='center', fontsize=6.5, color=C_GRAY, fontfamily='monospace', zorder=3)

    # Mismatch badge
    badge = FancyBboxPatch((7.8, 0.12), 1.6, 0.35,
                           boxstyle="round,pad=0,rounding_size=0.06",
                           facecolor=C_RED, edgecolor=C_RED, linewidth=1, zorder=3)
    ax.add_patch(badge)
    ax.text(8.6, 0.30, 'MISMATCH → Risk ↑', ha='center', va='center',
            fontsize=5.5, color='white', fontweight='bold', zorder=4)

    # Arrow connecting
    draw_arrow_v(ax, 3.9, 2.0, 1.05, color='#b7791f')

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_geospatial_pipeline.png'))
    plt.close()
    print('  [✓] Figure 2: Geospatial Pipeline')


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Audio Feature Extraction (KEEP — user approved)
# ═══════════════════════════════════════════════════════════════
def generate_fig3():
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.5))
    np.random.seed(42)

    # (a) Pitch Contour
    ax = axes[0, 0]
    t = np.linspace(0, 2, 500)
    normal_pitch = 120 + 5 * np.sin(2*np.pi*0.5*t) + np.random.normal(0, 2, 500)
    stressed_pitch = 140 + 25 * np.sin(2*np.pi*1.5*t) + np.random.normal(0, 12, 500)
    stressed_pitch = np.clip(stressed_pitch, 80, 220)
    ax.plot(t, normal_pitch, color=C_BLUE, lw=1.0, label='Normal (σ²=18 Hz²)')
    ax.plot(t, stressed_pitch, color=C_CORAL, lw=1.0, alpha=0.85, label='Stressed (σ²=42 Hz²)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('F0 (Hz)')
    ax.set_title('(a) Pitch Contour (F0)', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9); ax.set_ylim(60, 240); ax.grid(True, alpha=0.3)

    # (b) Jitter
    ax = axes[0, 1]
    cycles_n = np.cumsum(1.0 / (120 + np.random.normal(0, 0.8, 30)))
    cycles_s = np.cumsum(1.0 / (140 + np.random.normal(0, 6, 30)))
    periods_n = np.diff(cycles_n) * 1000
    periods_s = np.diff(cycles_s) * 1000
    ax.plot(range(len(periods_n)), periods_n, 'o-', color=C_BLUE, ms=3, lw=0.8,
            label=f'Normal (jitter={np.std(periods_n):.2f}ms)')
    ax.plot(range(len(periods_s)), periods_s, 's-', color=C_CORAL, ms=3, lw=0.8,
            label=f'Stressed (jitter={np.std(periods_s):.2f}ms)')
    ax.set_xlabel('Cycle Number'); ax.set_ylabel('Period (ms)')
    ax.set_title('(b) Jitter (F0 Perturbation)', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9); ax.grid(True, alpha=0.3)

    # (c) Shimmer
    ax = axes[1, 0]
    amp_n = 0.5 + np.random.normal(0, 0.01, 30)
    amp_s = 0.5 + np.random.normal(0, 0.06, 30)
    ax.bar(range(30), amp_n, width=0.4, color=C_BLUE, alpha=0.7, label='Normal (2.1%)')
    ax.bar(np.arange(30)+0.4, amp_s, width=0.4, color=C_CORAL, alpha=0.7, label='Stressed (8.7%)')
    ax.set_xlabel('Cycle Number'); ax.set_ylabel('Amplitude')
    ax.set_title('(c) Shimmer (Amplitude Perturbation)', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9); ax.grid(True, alpha=0.3, axis='y')

    # (d) Pause Rate
    ax = axes[1, 1]
    np.random.seed(42)
    normal_segs = np.ones(100); normal_segs[np.random.choice(100, 8, replace=False)] = 0
    stress_segs = np.ones(100); stress_segs[np.random.choice(100, 22, replace=False)] = 0
    ax.fill_between(range(100), 1.05, 1.95, where=normal_segs==1, color=C_BLUE, alpha=0.6, step='mid')
    ax.fill_between(range(100), 1.05, 1.95, where=normal_segs==0, color=C_GRAY_LT, alpha=0.5, step='mid')
    ax.fill_between(range(100), 0.05, 0.95, where=stress_segs==1, color=C_CORAL, alpha=0.6, step='mid')
    ax.fill_between(range(100), 0.05, 0.95, where=stress_segs==0, color=C_GRAY_LT, alpha=0.5, step='mid')
    ax.set_xlabel('Time Segments'); ax.set_ylabel('')
    ax.set_title('(d) Pause Rate (Speech Activity)', fontweight='bold')
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(['Stressed\n(22% pauses)', 'Normal\n(8% pauses)'])
    ax.set_ylim(-0.2, 2.2); ax.grid(True, alpha=0.3, axis='x')
    s_patch = mpatches.Patch(color=C_BLUE, alpha=0.6, label='Speech')
    p_patch = mpatches.Patch(color=C_GRAY_LT, alpha=0.5, label='Silence')
    ax.legend(handles=[s_patch, p_patch], loc='lower right', framealpha=0.9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_audio_features.png'))
    plt.close()
    print('  [✓] Figure 3: Audio Feature Extraction')


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Multimodal Fusion Model
# ═══════════════════════════════════════════════════════════════
def generate_fig4():
    fig, ax = plt.subplots(figsize=(7.16, 4.2))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.8, 6.5)
    ax.axis('off')

    # ── Left: Feature Vector ──
    ax.text(1.25, 6.2, 'Feature Vector (6D)', ha='center', fontsize=9,
            fontweight='bold', color=C_NAVY)

    features = [
        ('geo_shell_risk',        'Geospatial', C_TEAL_LIGHT, C_TEAL),
        ('audio_jitter_zscore',   'Audio',      C_LIGHT_BLUE, C_DARK_BLUE),
        ('audio_shimmer_zscore',  'Audio',      C_LIGHT_BLUE, C_DARK_BLUE),
        ('audio_pitch_variance',  'Audio',      C_LIGHT_BLUE, C_DARK_BLUE),
        ('audio_pause_rate',      'Audio',      C_LIGHT_BLUE, C_DARK_BLUE),
        ('text_semantic_evasion', 'Linguistic',  C_PURP_LIGHT, C_PURPLE),
    ]

    for i, (name, modality, fill, border) in enumerate(features):
        y = 5.6 - i * 0.8
        draw_box(ax, 0.0, y - 0.25, 2.5, 0.55,
                 name, None, fill=fill, border=border, text_color=border,
                 fontsize=5.8, lw=1.2)
        # Modality label
        ax.text(2.65, y, modality, ha='left', va='center', fontsize=5,
                color=C_SLATE, fontstyle='italic', zorder=3)
        # Arrow to model
        draw_arrow_diag(ax, 2.5, y, 4.5, 3.0, color=border, lw=0.8)

    # ── Center: GradientBoosting ──
    draw_box(ax, 4.5, 1.5, 3.0, 3.0,
             'GradientBoosting', None,
             fill=C_GRAY_LT, border=C_NAVY, text_color=C_NAVY, fontsize=10, lw=2)

    params = [
        'n_estimators = 100',
        'learning_rate = 0.05',
        'max_depth = 2',
        '',
        '5-Fold Stratified CV',
        'StandardScaler per fold',
    ]
    for i, p in enumerate(params):
        weight = 'bold' if i >= 4 else 'normal'
        color = C_TEAL if i >= 4 else C_GRAY
        ax.text(6.0, 4.0 - i * 0.33, p, ha='center', va='center',
                fontsize=5.8, color=color, fontweight=weight,
                fontfamily='monospace', zorder=3)

    # Decision tree icons (simplified)
    for j in range(3):
        tx = 5.0 + j * 0.9
        ty = 2.1
        # Simple tree: root → 2 leaves
        ax.plot([tx, tx], [ty, ty+0.3], color=C_NAVY, lw=0.8, zorder=3)
        ax.plot([tx, tx-0.2], [ty+0.3, ty+0.5], color=C_NAVY, lw=0.8, zorder=3)
        ax.plot([tx, tx+0.2], [ty+0.3, ty+0.5], color=C_NAVY, lw=0.8, zorder=3)
        ax.plot(tx, ty+0.5, 'o', color=C_NAVY, ms=2.5, zorder=3)
        ax.plot(tx-0.2, ty+0.5, 's', color=C_GREEN, ms=2, zorder=3)
        ax.plot(tx+0.2, ty+0.5, 's', color=C_RED, ms=2, zorder=3)
        if j < 2:
            ax.text(tx + 0.42, ty + 0.25, '+', fontsize=8, color=C_NAVY,
                    ha='center', va='center', fontweight='bold', zorder=3)

    # ── Arrow to output ──
    draw_arrow_h(ax, 7.5, 8.2, 3.0, color=C_NAVY, lw=1.5)

    # ── Right: Output ──
    ax.text(9.75, 6.2, 'CRDI Output', ha='center', fontsize=9,
            fontweight='bold', color=C_NAVY)

    # P(fraud) label
    ax.text(9.75, 5.75, 'CRDI = P(fraud) × 100', ha='center', fontsize=7, color=C_SLATE)

    # Risk bands
    bands = [
        ('CREDIBLE',   '0–30',  C_GREEN_LT, C_GREEN),
        ('WATCHLIST',  '30–50', C_YELLOW_LT, '#b7791f'),
        ('SUSPICIOUS', '50–70', C_ORANGE_LT, C_ORANGE),
        ('HIGH_RISK',  '70–85', C_RED_LT,    C_RED),
        ('CRITICAL',   '85–100',C_RED_LT,    '#9b2c2c'),
    ]
    for i, (band, rng, fill, border) in enumerate(bands):
        y = 5.05 - i * 0.7
        draw_box(ax, 8.3, y, 2.9, 0.55,
                 f'{band}  ({rng})', None,
                 fill=fill, border=border, text_color=border,
                 fontsize=7, lw=1.2)

    # ── Bottom: Performance badge ──
    draw_box(ax, 2.5, -0.55, 6.5, 0.6,
             'F1 = 0.905 ± 0.052    |    AUC-ROC = 0.976 ± 0.015    |    N = 276', None,
             fill=C_LIGHT_BLUE, border=C_NAVY, text_color=C_NAVY,
             fontsize=7.5, lw=1.5)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_fusion_model.png'))
    plt.close()
    print('  [✓] Figure 4: Fusion Model Architecture')


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: SHAP Feature Importance (KEEP — user approved)
# ═══════════════════════════════════════════════════════════════
def generate_fig5():
    fig, ax = plt.subplots(figsize=(7.16, 2.8))

    features = [
        'audio_pitch_variance', 'audio_shimmer_zscore',
        'audio_pause_rate', 'audio_jitter_zscore',
        'geo_shell_risk', 'text_semantic_evasion',
    ]
    shap_vals = [0.0031, 0.0304, 0.1599, 0.2952, 0.5276, 1.6477]
    pcts = [0.1, 1.1, 6.0, 11.1, 19.8, 61.9]
    colors = ['#CBD5E0', '#A0AEC0', '#63B3ED', C_BLUE, C_DARK_BLUE, C_NAVY]

    bars = ax.barh(range(len(features)), shap_vals, color=colors,
                   edgecolor='black', linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontfamily='monospace', fontsize=8)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=9)
    ax.set_title('Global SHAP Feature Importance', fontsize=10, fontweight='bold', color=C_NAVY)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                f'{pct}%', ha='left', va='center', fontsize=8, fontweight='bold', color=C_SLATE)

    ax.set_xlim(0, 2.0)
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_shap_importance.png'))
    plt.close()
    print('  [✓] Figure 5: SHAP Feature Importance')


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 60)
    print('  Generating Publication Figures (300 DPI)')
    print(f'  Output: {OUTPUT_DIR}/')
    print('=' * 60)

    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()

    print()
    print(f'  ✓ All 5 figures saved at 300 DPI')
    print('=' * 60)
