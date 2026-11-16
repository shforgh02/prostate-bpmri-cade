# Figure 4 (published number; file stem Figure5_case_examples): four example cases chosen by rule, not by eye.
# From B6_targets_final_qc.csv, single-target examinations with a sampled target:
#   a score 3, all three models agree, target csPCa; b score 3, no model agrees, target benign;
#   c score 4 or 5, all three agree, target csPCa;  d score 3, no model agrees, target csPCa.
# In each category the examination whose target volume is the median of the category (closest to the median; ties by Session_ID).
# Each row: T2-weighted, ADC and high-b images at the axial slice of the target center; target outline; candidate centers of the three models
# within 5 mm of the slice (model colors and marker shapes); core intersections with the slice (filled = csPCa, open circle = benign, open triangle = grade group 1).
#   python analysis/D7_case_examples.py
import os
import numpy as np, pandas as pd, nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

O, FIG = 'outputs/', 'figures/'
SD = 'results/tables/'
IMG = os.environ.get('PROSTATE_DATA_ROOT', '../..') + '/MRI_US_Biopsy/Script/output/standardized_ucla_sessions_new/'
MODELS = [('3D_SegResNet', 'SegResNet', '#2a78d6'), ('nnUNet_official', 'nnU-Net', '#eb6834'), ('ProViCNet', 'ProViCNet', '#1baf7a')]
MARK = {'3D_SegResNet': '+', 'nnUNet_official': 'x', 'ProViCNet': '1'}          # one shape per model so that overlapping candidates stay visible
CC = ['Bx Base X (MRI Coord)', 'Bx Base Y (MRI Coord)', 'Bx Base Z (MRI Coord)', 'Bx Tip X (MRI Coord)', 'Bx Tip Y (MRI Coord)', 'Bx Tip Z (MRI Coord)']

T = pd.read_csv(O + 'B6_targets_final_qc.csv')
T = T[(T.n_targets == 1) & (T.path != 'unsampled')].copy()
T['n_agree'] = T[[f'agree_{m}' for m, _, _ in MODELS]].sum(1)
CAT = [('a', 'Score 3, all three models agree, csPCa', (T.score == 3) & (T.n_agree == 3) & (T.path == 'csPCa')),
       ('b', 'Score 3, no model agrees, benign', (T.score == 3) & (T.n_agree == 0) & (T.path == 'benign')),
       ('c', 'Score 4-5, all three models agree, csPCa', T.score.isin([4, 5]) & (T.n_agree == 3) & (T.path == 'csPCa')),
       ('d', 'Score 3, no model agrees, csPCa', (T.score == 3) & (T.n_agree == 0) & (T.path == 'csPCa'))]
sel = []
for key, lab, m in CAT:
    q = T[m].sort_values('Session_ID')
    med = q.vol_cc.median()
    r = q.iloc[(q.vol_cc - med).abs().to_numpy().argmin()]
    sel.append((key, lab, r, len(q), med))
    print(f'{key}: {lab} | category n = {len(q)}, median target volume {med:.2f} mL -> {r.Session_ID} (target volume {r.vol_cc:.2f} mL)')

cand = pd.read_csv(O + 'CANDIDATES_3MODELS.csv', low_memory=False)
cand = cand[(cand.Extractor == 'Official_RGA_Dynamic') & (cand.Is_Canonical == 1) & (cand.Candidate_Rank <= 3)]
cores = pd.read_csv(SD + 'CORES_REFERENCE.csv', low_memory=False)


def lps_to_vox(aff, lps):
    ras = np.asarray(lps, float) * np.array([-1.0, -1.0, 1.0])
    return (np.linalg.inv(aff) @ np.append(ras, 1.0))[:3]


fig, axes = plt.subplots(4, 3, figsize=(7.0, 9.6))
for row, (key, lab, r, ncat, med) in enumerate(sel):
    s = r.Session_ID
    d = IMG + s + '/'
    t2 = nib.load(d + 't2_norm.nii.gz'); aff = t2.affine
    vols = [np.asarray(t2.dataobj, float), np.asarray(nib.load(d + 'adc_norm.nii.gz').dataobj, float), np.asarray(nib.load(d + 'dwi_norm.nii.gz').dataobj, float)]
    tm = np.asarray(nib.load(d + 'target_1_mask.nii.gz').dataobj) > 0
    pm = np.asarray(nib.load(d + 'prostate_mask.nii.gz').dataobj) > 0
    com = np.array(ndimage.center_of_mass(tm))
    k = int(round(com[2]))
    ii, jj = np.where(pm.any(2))
    i0, i1, j0, j1 = max(ii.min() - 12, 0), min(ii.max() + 12, tm.shape[0]), max(jj.min() - 12, 0), min(jj.max() + 12, tm.shape[1])
    # candidates within 5 mm of the slice
    cpts = []
    for m, nm, col in MODELS:
        for _, c in cand[(cand.Model == m) & (cand.Session_ID == s)].iterrows():
            v = lps_to_vox(aff, c[['CoM_LPS_X_mm', 'CoM_LPS_Y_mm', 'CoM_LPS_Z_mm']].to_numpy(float))
            if abs(v[2] - k) * abs(aff[2, 2]) <= 5:
                cpts.append((v[0], v[1], col, MARK[m]))
    # core intersections with the slice plane
    kpts = []
    for _, c in cores[(cores.Session_ID_Matched == s) & (cores.Has_Valid_Physical_Trajectory == 1)].iterrows():
        a, b = lps_to_vox(aff, c[CC[:3]].to_numpy(float)), lps_to_vox(aff, c[CC[3:]].to_numpy(float))
        if (a[2] - k) * (b[2] - k) <= 0 and a[2] != b[2]:
            t = (k - a[2]) / (b[2] - a[2]); p = a + t * (b - a)
            kind = 'cs' if c.Is_csPCa == 1 else ('gg1' if c.Is_GG1_Indolent == 1 else 'ben')
            kpts.append((p[0], p[1], kind))
    for col_i, (vol, title) in enumerate(zip(vols, ('T2-weighted', 'ADC', 'High-b DWI'))):
        ax = axes[row, col_i]
        sl = vol[i0:i1, j0:j1, k].T                       # rows = posterior direction (anterior at top), columns = patient left to the right
        lo, hi = np.percentile(sl[np.isfinite(sl)], [1, 99])
        ax.imshow(sl, cmap='gray', vmin=lo, vmax=hi, origin='upper', interpolation='bilinear')
        ax.contour(tm[i0:i1, j0:j1, k].T.astype(float), levels=[0.5], colors='#f2c94c', linewidths=1.0)
        ax.contour(pm[i0:i1, j0:j1, k].T.astype(float), levels=[0.5], colors='white', linewidths=0.5, alpha=0.6)
        for x, y, kind in kpts:
            if kind == 'cs':
                ax.plot(x - i0, y - j0, 'o', ms=4.5, color='#e5484d', mec='white', mew=0.6, zorder=4)
            elif kind == 'gg1':
                ax.plot(x - i0, y - j0, '^', ms=4.5, mfc='none', mec='#f2c94c', mew=0.9, zorder=4)
            else:
                ax.plot(x - i0, y - j0, 'o', ms=4.5, mfc='none', mec='white', mew=0.9, zorder=4)
        for x, y, col, mk in cpts:                       # candidates drawn last so that cores do not hide them
            ax.plot(x - i0, y - j0, marker=mk, ms=9, mew=2.0, color=col, ls='', zorder=6)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(title, fontsize=8)
        if col_i == 0:
            shown = f'Score {int(r.score)}' + lab[lab.index(','):]      # row label with the score of the selected target
            ax.set_ylabel(f'{key}  {shown}', fontsize=7, rotation=90, labelpad=6)
leg = [plt.Line2D([], [], marker=MARK[m], ls='', ms=8, mew=2, color=c) for m, _, c in MODELS] + \
      [plt.Line2D([], [], color='#f2c94c', lw=1.2), plt.Line2D([], [], color='#8a8a85', lw=0.8), plt.Line2D([], [], marker='o', ls='', color='#e5484d', ms=5),
       plt.Line2D([], [], marker='o', ls='', mfc='none', mec='#5f5e5a', ms=5), plt.Line2D([], [], marker='^', ls='', mfc='none', mec='#b08800', ms=5)]
fig.legend(leg, [n for _, n, _ in MODELS] + ['Radiologist target', 'Prostate contour', 'csPCa core', 'Benign core', 'Grade group 1 core'], loc='lower center', ncol=4,
           fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.035))
fig.tight_layout(h_pad=0.4, w_pad=0.3)
fig.savefig(FIG + 'Figure5_case_examples.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(FIG + 'Figure5_case_examples.pdf', bbox_inches='tight', facecolor='white')
pd.DataFrame([dict(panel=k, category=l, Session_ID=r.Session_ID, target_volume_ml=r.vol_cc, category_n=n, category_median_volume_ml=m)
              for k, l, r, n, m in sel]).to_csv(O + 'D7_case_selection.csv', index=False)
print('saved', FIG + 'Figure5_case_examples')
