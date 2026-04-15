"""
Fig 1 — Multimodal Corporate Fraud Framework Architecture
Implements the revised wide-layout user specifications.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# A wider figure size to accommodate the horizontal layout
fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
ax.set_xlim(0, 160); ax.set_ylim(-30, 115)
ax.set_aspect('equal')
ax.axis('off')

plt.rcParams.update({'font.family': 'DejaVu Sans'})

# ── Colors ────────────────────────────────────────────────────────
C_GEO    = '#48B8A0';  C_GEO_L  = '#D4EFE8'
C_AUD    = '#5B7FA6';  C_AUD_L  = '#D6E2ED'
C_LNG    = '#8E6BAD';  C_LNG_L  = '#E3DBED'
C_GREY   = '#7F8C8D';  C_GREY_L = '#EAEDED'
C_DARK   = '#34495E';  C_TEXT   = '#2C3E50'
C_FUSE   = '#607D8B';  C_FUSE_L = '#CFD8DC'
C_CRDI   = '#D35400';  C_CRDI_L = '#FADBD8'
C_AUDIT  = '#E67E22';  C_AUDIT_L= '#FDEBD0'
C_EXP    = '#9C89B8';  C_EXP_L  = '#EBE5F2'

def box(ax, x, y, w, h, fc, ec, lw=1.5, pad=0.5, shadow=True, z=2):
    if shadow:
        s = FancyBboxPatch((x+0.5, y-0.5), w, h, boxstyle=f"round,pad={pad}",
                           fc='#00000018', ec='none', lw=0, zorder=1)
        ax.add_patch(s)
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)

def t(ax, x, y, s, fs=8, fw='normal', c=C_TEXT, ha='center', va='center', z=10, **kw):
    return ax.text(x, y, s, fontsize=fs, fontweight=fw, color=c, ha=ha, va=va, zorder=z, **kw)

def arrow(ax, src, dst, color=C_GREY, lw=1.5, style='-|>', rad=0.0):
    if rad == 0.0:
        cs = "arc3"
    else:
        cs = f"arc3,rad={rad}"
    ap = FancyArrowPatch(src, dst, connectionstyle=cs, arrowstyle=style, color=color, lw=lw, zorder=5)
    ax.add_patch(ap)

# ════════════════════════════════════════════════════════════════
#  1. HEADER & TOP (Validation, Profile, SEC)
# ════════════════════════════════════════════════════════════════
# Title
t(ax, 5, 110, "Figure 1: End-to-End CRDI Framework: Multimodal Corporate Fraud Risk Estimation via Geospatial,\nAuditory, and Linguistic Signal Fusion",
  fs=14, fw='bold', ha='left', va='top')

# Data Validation
box(ax, 45, 93, 70, 7, C_GREY_L, C_GREY)
t(ax, 80, 98, 'Data Validation', fs=10, fw='bold')
t(ax, 80, 95.2, '• GPS coordinates verified ✓     • Audio file readable ✓     • Transcript non-empty ✓', fs=8.5)

# Company Profile
box(ax, 60, 80, 40, 8, C_GREY_L, C_GREY)
t(ax, 80, 85.5, 'Company Profile', fs=10, fw='bold')
t(ax, 80, 82.5, '• Name   • GPS   • Audio   • Transcript', fs=8.5)

# SEC AAER
box(ax, 10, 80, 25, 8, '#BDC3C7', '#7F8C8D')
t(ax, 22.5, 84, 'SEC AAER\nDatabase', fs=10, fw='bold')

# Top arrows
arrow(ax, (80, 93), (80, 88.5))
ap = FancyArrowPatch((36, 84), (59, 84), connectionstyle="arc3", arrowstyle="-|>", 
                     color=C_GREY, lw=1.5, linestyle='dashed', zorder=5)
ax.add_patch(ap)
t(ax, 47.5, 85.8, 'fraud ground truth labels', fs=7, style='italic', c=C_DARK)

# Arrows from profile to modules
arrow(ax, (75, 79.5), (35, 75.5), color=C_GEO, rad=0.2)
arrow(ax, (80, 79.5), (87.5, 75.5), color=C_AUD, rad=0.1)
arrow(ax, (85, 79.5), (127.5, 75.5), color=C_LNG, rad=-0.2)

# ════════════════════════════════════════════════════════════════
#  2. MODALITY MODULES
# ════════════════════════════════════════════════════════════════

# ── Geospatial Module ──
box(ax, 5, 20, 60, 54, C_GEO_L, C_GEO)
t(ax, 35, 71, 'Geospatial Module (Integrated Detailed View)', fs=10, fw='bold', c='#1A7A5C')

# Inner Geo boxes
box(ax, 8, 56, 17, 12, 'white', C_GEO, lw=1, pad=0)
t(ax, 16.5, 65, 'GPS Input', fs=8, fw='bold')
t(ax, 16.5, 60, 'Lat. 42.9628N\nLong. 200.6918\nCompany Name', fs=7)

box(ax, 30, 56, 32, 12, 'white', C_GEO, lw=1, pad=0)
t(ax, 46, 65, 'Google Street View API', fs=8, fw='bold')
t(ax, 46, 60, '[Street imagery retrieval]', fs=7, style='italic', c=C_GREY)

box(ax, 8, 36, 25, 16, 'white', C_GEO, lw=1, pad=0)
box(ax, 8, 48.5, 25, 3.5, C_GEO, C_GEO, lw=0, pad=0, shadow=False)
t(ax, 20.5, 50.2, 'Places365 CNN', fs=8, fw='bold', c='white')
t(ax, 20.5, 47, 'ResNet-18', fs=7.5, fw='bold')
t(ax, 9, 44, 'office_building', fs=7, ha='left')
t(ax, 32, 44, '→ 0.05', fs=7, ha='right', c='#27AE60', fw='bold')
t(ax, 9, 41, 'residential', fs=7, ha='left')
t(ax, 32, 41, '→ 0.70', fs=7, ha='right', c='#E67E22', fw='bold')
t(ax, 9, 38, 'vacant_lot', fs=7, ha='left')
t(ax, 32, 38, '→ 0.90', fs=7, ha='right', c='#C0392B', fw='bold')

box(ax, 38, 36, 24, 16, 'white', C_GEO, lw=1, pad=0)
box(ax, 38, 48.5, 24, 3.5, '#E67E22', '#E67E22', lw=0, pad=0, shadow=False)
t(ax, 50, 50.2, 'Shell Risk Score', fs=8, fw='bold', c='white')
t(ax, 50, 46.5, '(geo_shell_risk)', fs=6.5)
# gradient placeholder
grad = FancyBboxPatch((42, 40), 16, 4, boxstyle="round,pad=0", fc='#A3E4D7', ec='none')
ax.add_patch(grad)
t(ax, 42, 38.5, '0.0', fs=6); t(ax, 58, 38.5, '1.0', fs=6)

box(ax, 8, 23, 54, 9, 'white', C_GEO, lw=1, pad=0)
box(ax, 8, 28, 54, 4, '#F5B041', '#F5B041', lw=0, pad=0, shadow=False)
t(ax, 35, 30, 'Entity Name Verification', fs=8, fw='bold', c=C_TEXT)
t(ax, 10, 25.5, 'Compare company name vs.\nGoogle Places result', fs=7, ha='left')
box(ax, 46, 24, 14, 3, '#F5B041', '#E67E22', lw=0.5, pad=0)
t(ax, 53, 25.5, 'MISMATCH', fs=6.5, fw='bold')

# Geo arrows
arrow(ax, (25.5, 62), (29.5, 62), color=C_GEO, lw=1)
arrow(ax, (46, 55.5), (20.5, 52.5), color=C_GEO, lw=1)
arrow(ax, (33.5, 44), (37.5, 44), color=C_GEO, lw=1)

# ── Audio Module ──
box(ax, 70, 35, 35, 39, C_AUD_L, C_AUD)
t(ax, 87.5, 71, 'Audio DSP Module', fs=10, fw='bold', c=C_DARK)
box(ax, 73, 61, 29, 6.5, 'white', C_AUD, pad=0)
t(ax, 87.5, 64.2, 'Feature Extraction: Librosa DSP', fs=8)
box(ax, 73, 52, 29, 6.5, 'white', C_AUD, pad=0)
t(ax, 87.5, 55.2, 'Prosodic Analysis: Pitch, Pauses', fs=8)
box(ax, 73, 43, 29, 6.5, 'white', C_AUD, pad=0)
t(ax, 87.5, 46.2, 'Spectral Features: Jitter, Shimmer', fs=8)

# ── Linguistic Module ──
box(ax, 110, 35, 35, 39, C_LNG_L, C_LNG)
t(ax, 127.5, 71, 'Linguistic Module', fs=10, fw='bold', c='#4A235A')
box(ax, 113, 61, 29, 6.5, 'white', C_LNG, pad=0)
t(ax, 127.5, 64.2, 'Language Modeling: Llama 3.3 70B', fs=8)
box(ax, 113, 52, 29, 6.5, 'white', C_LNG, pad=0)
t(ax, 127.5, 55.2, 'Reasoning: Chain-of-Thought', fs=8)
box(ax, 113, 43, 29, 6.5, 'white', C_LNG, pad=0)
t(ax, 127.5, 46.2, 'Evasion Detection: Semantic Evasion', fs=8)

# ════════════════════════════════════════════════════════════════
#  3. FUSION LAYER & SCALER
# ════════════════════════════════════════════════════════════════

box(ax, 70, 14, 45, 14, C_FUSE_L, C_FUSE, z=10) # Bring box above loop lines
t(ax, 92.5, 25, 'ML Fusion Engine', fs=10, fw='bold', z=11)
t(ax, 92.5, 21.5, 'GradientBoosting Classifier', fs=9, z=11)
t(ax, 92.5, 17.5, '6 features, F1=0.905', fs=8, c=C_DARK, z=11)

box(ax, 130, 16, 25, 10, '#EAEDED', '#BDC3C7')
t(ax, 142.5, 23, 'Standard Scaler', fs=9, fw='bold')
t(ax, 142.5, 19, 'Z-score normalization\n(per fold)', fs=7)

# Arrow from Scaler into ML Fusion
arrow(ax, (130, 21), (115.5, 21), color=C_TEXT, lw=1.5)
t(ax, 122.5, 22.5, 'normalized features', fs=7)

# 5-Fold Stratified CV Loop Wrap around Fusion Engine
# Draw top arc (zorder 5 ensures it goes UNDER the text and box a bit)
arc1 = FancyArrowPatch((65, 21), (120, 21), connectionstyle="arc3,rad=-0.5", 
                       arrowstyle="->", color=C_FUSE, lw=1.5, zorder=5)
ax.add_patch(arc1)
# Draw bottom arc
arc2 = FancyArrowPatch((120, 21), (65, 21), connectionstyle="arc3,rad=-0.5", 
                       arrowstyle="->", color=C_FUSE, lw=1.5, zorder=5)
ax.add_patch(arc2)

# Text placed shifted right to avoid top arrows
box(ax, 115, 29, 36, 6, 'white', 'none', pad=0, shadow=False) # masking box
t(ax, 134, 32, '5-Fold Stratified CV\nk=5, fraud prevalence 18.1%', fs=7.5, fw='bold', c=C_FUSE)

# Output arrows from 3 modules to ML Fusion (zorder > arc to be visible)
ap_geo = FancyArrowPatch((65.5, 42), (76, 28.5), connectionstyle="arc3,rad=0.2", arrowstyle="-|>", color=C_GEO, lw=2, zorder=8)
ax.add_patch(ap_geo)
t(ax, 68, 37, 'geo_shell_risk', fs=8, c='#1A7A5C', fw='bold', z=12)

ap_aud = FancyArrowPatch((87.5, 34.5), (87.5, 28.5), connectionstyle="arc3", arrowstyle="-|>", color=C_AUD, lw=2, zorder=8)
ax.add_patch(ap_aud)
t(ax, 82, 31.5, '[j, s, v, p]', fs=8, c=C_AUD, fw='bold', z=12)

ap_lng = FancyArrowPatch((127.5, 34.5), (105, 28.5), connectionstyle="arc3,rad=-0.2", arrowstyle="-|>", color=C_LNG, lw=2, zorder=8)
ax.add_patch(ap_lng)
t(ax, 116, 36, 'ℓ (semantic evasion)', fs=8, c=C_LNG, fw='bold', z=12)

# ════════════════════════════════════════════════════════════════
#  4. EXPLAINABILITY & OUTPUTS
# ════════════════════════════════════════════════════════════════

arrow(ax, (92.5, 13.5), (92.5, 7), color=C_GREY)

box(ax, 70, -5, 55, 11.5, C_EXP_L, C_EXP)
t(ax, 97.5, 4, 'Explainability Layer', fs=10, fw='bold', c='#4A235A')
t(ax, 83.5, 0, 'SHAP TreeExplainer', fs=9)
t(ax, 111.5, 0, 'LLM Rationale\n(3 Llama calls)', fs=8)

arrow(ax, (92.5, -5.5), (92.5, -10), color=C_GREY)

# CRDI Score output (now with gauge instead of dots)
# Widened to 40 so labels definitely fit
box(ax, 72.5, -23, 40, 12.5, C_CRDI_L, C_CRDI)
t(ax, 92.5, -14, 'CRDI Score (0-100)', fs=10, fw='bold', c='#A04000')

# Color gauge inside CRDI Output
gw = 36; gh = 5; gx = 74.5; gy = -21
bands = [
    (0.24, '#27AE60', 'CREDIBLE'),
    (0.20, '#7DCEA0', 'WATCHLIST'),
    (0.21, '#F4D03F', 'SUSPICIOUS'),
    (0.18, '#E67E22', 'HIGH RISK'),
    (0.17, '#C0392B', 'CRITICAL')
]
cx = gx
for p, c, l in bands:
    bw = gw * p
    r = FancyBboxPatch((cx, gy), bw, gh, boxstyle="square,pad=0", fc=c, ec='none', zorder=5)
    ax.add_patch(r)
    # tiny text inside
    ax.text(cx + bw/2, gy + gh/2, l, color='white', weight='bold', fontsize=4.2, 
            ha='center', va='center', zorder=6)
    cx += bw

# Audit Report
arrow(ax, (112.5, -16), (119.5, -16), color=C_AUDIT, lw=1.5)

box(ax, 120, -26, 33, 16, C_AUDIT_L, C_AUDIT)
t(ax, 136.5, -13, 'Audit Report Output', fs=10, fw='bold', c='#A04000')
t(ax, 123, -17.5, '• CRDI Score (0-100)', fs=8, ha='left')
t(ax, 123, -20.5, '• Risk Band label', fs=8, ha='left')
t(ax, 123, -23.5, '• Natural language rationale', fs=8, ha='left')


# ── Save ────────────────────────────────────────────────────────
fig.savefig('fig1_methodology.png', dpi=300, bbox_inches='tight', pad_inches=0.15, facecolor='white')
fig.savefig('fig1_methodology.pdf', dpi=300, bbox_inches='tight', pad_inches=0.15, facecolor='white')
plt.close()
print("✓ fig1_methodology wide version saved successfully")
