# Figures 2, 3 and 5 (published numbers; file stems: Figure2_localization_vs_chance, Figure3_agreement, Figure4_offtarget_cores) and Figure S1,
# plus the Table S1 CSV. matplotlib only. Numbers are read from outputs/ files; nothing is typed in by hand.
#   python analysis/D1_figures.py
# Figures -> figures/*.png (300 dpi) and *.pdf.  Table S1 -> tables/TableS1_exclusions.csv
import os, re, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

O = 'outputs/'
FIG, TAB = 'figures/', 'tables/'
SD = 'results/tables/'
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# One colour per model, fixed order (validated: dataviz validate_palette.js, light mode, all checks pass).
MODELS = [('3D_SegResNet', 'SegResNet', '#2a78d6'),            # training-set sizes are given in Table 2 footnotes, not on axes
          ('nnUNet_official', 'nnU-Net', '#eb6834'),
          ('ProViCNet', 'ProViCNet', '#1baf7a')]
KEYS = [m[0] for m in MODELS]
NAME = {k: n for k, n, _ in MODELS}
COL = {k: c for k, _, c in MODELS}
INK, INK2, GRID = '#1f1f1e', '#5f5e5a', '#e6e5e0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42})


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return ''
    if round(p, 3) < 0.001:
        return 'P < .001'
    v = f'{p:.3f}' if p < 0.01 else f'{p:.2f}'
    return 'P = ' + v.lstrip('0')


def pct(x, d=0):
    return f'{100 * x:.{d}f}%'


def save(fig, name):
    fig.savefig(FIG + name + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG + name + '.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', FIG + name)


def legend_models(ax, loc='upper right', **kw):
    hs = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in KEYS]
    ax.legend(hs, [NAME[k] for k in KEYS], loc=loc, fontsize=7, handlelength=1.0, handleheight=0.8, **kw)


# ---------------------------------------------------------------- Figure 2: localization against chance (A1, 10 mm)
EP = pd.read_csv(SD + 'EXAMINATION_LEVEL_ENDPOINTS.csv')
EP = EP[~EP.Session_ID.isin(pd.read_csv(O + 'EXCLUDED_SESSIONS_v2.csv').Session_ID)]
EV = EP[(EP.Is_csPCa == 1) & (EP.Has_Valid_3D_Trajectory == 1)]
RAD_HIT, RAD_N = int(EV.Human_Target_Geometry_H.sum()), len(EV)
a1 = pd.read_csv(O + 'A1_results_final_qc.csv')
a1 = a1[a1.tol_mm == 10.0]
GROUPS = [('all csPCa', 'All cancers'), ('rad target HIT', 'Localized by\nradiologist target'), ('rad target MISS', 'Missed by\nradiologist target')]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
w, gap = 0.26, 0.03
for gi, (sub, lab) in enumerate(GROUPS):
    for mi, k in enumerate(KEYS):
        r = a1[(a1.model == k) & (a1.subset == sub)].iloc[0]
        x = gi + (mi - 1) * (w + gap)
        obs, ch = r.observed / r.n, r.perm_expected / r.n
        ax.bar(x, obs, w, color=COL[k], zorder=2)
        ax.text(x, obs + 0.015, pct(obs), ha='center', va='bottom', fontsize=7, color=INK)
        ax.plot([x - w / 2, x + w / 2], [ch, ch], color=INK, lw=1.6, zorder=3, solid_capstyle='butt')
        ax.text(x, ch - 0.02, pct(ch), ha='center', va='top', fontsize=6.5, color='white', zorder=4, fontweight='bold')
    n = int(a1[(a1.model == KEYS[0]) & (a1.subset == sub)].n.iloc[0])
    ax.text(gi, -0.155, f'n = {n}', ha='center', va='top', fontsize=7, color=INK2, transform=ax.get_xaxis_transform())
xr = RAD_HIT / RAD_N
xb = -2 * (w + gap)                                  # radiologist reference bar to the left of the 'All cancers' group
ax.bar(xb, xr, w, color='#b4b2a9', zorder=2)
ax.text(xb, xr + 0.015, pct(xr), ha='center', va='bottom', fontsize=7, color=INK)
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([g[1] for g in GROUPS], fontsize=8)
ax.set_xlim(-0.8, 2.5)
ax.set_ylim(0, 1.08)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticklabels([pct(v) for v in np.arange(0, 1.01, 0.2)])
ax.set_ylabel('Cancers localized by a top-3 candidate\n(within 10 mm of a positive core)')
ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
hs = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in KEYS] + [plt.Line2D([], [], color=INK, lw=1.6), plt.Rectangle((0, 0), 1, 1, color='#b4b2a9')]
ax.legend(hs, [NAME[k] for k in KEYS] + ['Expected by chance', f'Radiologist targets ({RAD_HIT}/{RAD_N})'], loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=5,
          fontsize=7, handlelength=1.4)
save(fig, 'Figure2_localization_vs_chance')

# ---------------------------------------------------------------- Figure 3: agreement with radiologist targets (B6)
b6 = pd.read_csv(O + 'B6_results_final_qc.csv')
tg = pd.read_csv(O + 'B6_targets_final_qc.csv')
tg = tg[(tg.path != 'unsampled') & tg.score.isin([3, 4, 5]) & np.isfinite(tg.lpsad)].copy()
tg['y'] = (tg.path == 'csPCa').astype(int)
tg['n_agree'] = tg[[f'agree_{k}' for k in KEYS]].sum(1)
fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.62, wspace=0.12)
axs = [fig.add_subplot(gs[0, 0])]
axs += [fig.add_subplot(gs[0, i], sharey=axs[0]) for i in (1, 2)]
for ax, sc in zip(axs, (3, 4, 5)):
    d = b6[b6.subset == f'score {sc}'].set_index('model')
    for i, k in enumerate(KEYS):
        r = d.loc[k]
        for j, (num, den) in enumerate(((r.agree_cspca, r.agree_n), (r.disagree_cspca, r.disagree_n))):
            x = i + (j - 0.5) * 0.36
            pr = num / den
            lo, hi = proportion_confint(int(num), int(den), method='wilson')
            if j == 0:
                ax.bar(x, pr, 0.33, color=COL[k], zorder=2)
            else:
                ax.bar(x, pr, 0.33, color='white', edgecolor=COL[k], lw=1.2, hatch='//////', zorder=2)
            ax.errorbar(x, pr, yerr=[[pr - lo], [hi - pr]], color=INK, lw=0.8, capsize=2, zorder=3)
            ax.text(x, -0.012 - 0.045 * j, f'{int(num)}/{int(den)}', ha='center', va='top', fontsize=5.6, color=INK2)
        ax.text(i, 1.09, f'OR {r.OR:.2f}\n({r.lo:.2f}–{r.hi:.2f})', ha='center', va='center', fontsize=6.0, color=INK, clip_on=False)
    ax.set_xticks(range(3))
    ax.set_xticklabels([NAME[k] for k in KEYS], fontsize=7)
    ax.tick_params(axis='x', pad=18, length=0)
    ax.set_ylim(0, 1.0)
    ax.set_title(('a  ' if sc == 3 else '') + f'Suspicion score {sc} (n = {int(d.n.iloc[0])})', fontsize=8, color=INK, loc='left', pad=24)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if sc != 3:
        plt.setp(ax.get_yticklabels(), visible=False)
axs[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: pct(v)))
axs[0].set_ylabel('Targets with csPCa')
hs = [plt.Rectangle((0, 0), 1, 1, color=INK2), plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=INK2, hatch='//////')]
axs[1].legend(hs, ['Candidate within 10 mm of the target center', 'No candidate within 10 mm'], loc='upper center', bbox_to_anchor=(0.5, -0.30),
              ncol=2, fontsize=7)
axb = fig.add_subplot(gs[1, :])
GREY = ['#d9d8d2', '#b4b2a9', '#888780', '#5f5e5a']
for si, sc in enumerate((3, 4, 5)):
    q = tg[tg.score == sc]
    for k in range(4):
        z = q[q.n_agree == k]
        num, den = int(z.y.sum()), len(z)
        pr = num / den
        lo, hi = proportion_confint(num, den, method='wilson')
        x = si + (k - 1.5) * 0.2
        axb.bar(x, pr, 0.18, color=GREY[k], zorder=2)
        axb.errorbar(x, pr, yerr=[[pr - lo], [hi - pr]], color=INK, lw=0.8, capsize=2, zorder=3)
        axb.text(x, -0.012, f'{num}/{den}', ha='center', va='top', fontsize=5.6, color=INK2)
axb.set_xticks(range(3))
axb.set_xticklabels([f'Suspicion score {s}' for s in (3, 4, 5)], fontsize=7.5)
axb.tick_params(axis='x', pad=11, length=0)
axb.set_ylim(0, 1.0)
axb.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: pct(v)))
axb.set_ylabel('Targets with csPCa')
axb.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
axb.set_axisbelow(True)
axb.set_title('b  Number of models agreeing with the target', fontsize=8, color=INK, loc='left')
axb.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in GREY], ['0 models', '1 model', '2 models', '3 models'], loc='upper left', ncol=4, fontsize=7)
save(fig, 'Figure3_agreement')

# ---------------------------------------------------------------- Figure 5 (published; stem Figure4_offtarget_cores): cores away from and near the targets (A2)
a2 = pd.read_csv(O + 'A2_results_final_qc.csv')
hol = pd.read_csv(O + 'C4_holm.csv'); hol = hol[hol.tag == 'final_qc'].set_index('model')      # panel a: Holm-adjusted P across the three models
ROWS = [('OFF-TARGET, 10 mm (main)', 'a  Systematic cores away from radiologist targets'),
        ('near radiologist target, 10 mm (positive control)', 'b  Systematic cores near radiologist targets')]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, (row, title) in zip(axs, ROWS):
    d = a2[a2.analysis == row].set_index('model')
    for i, k in enumerate(KEYS):
        r = d.loc[k]
        ax.add_patch(plt.Rectangle((i - 0.18, r.lo), 0.36, r.hi - r.lo, color='#d9d8d2', lw=0, zorder=1))
        ax.plot([i - 0.18, i + 0.18], [r.null_mean, r.null_mean], color=INK2, lw=1.0, zorder=2)
        ax.plot(i, r.value, 'o', ms=7, color=COL[k], mec='white', mew=1.2, zorder=3)
        ax.text(i + 0.22, r.value, f'{pct(r.value, 1)}\n({int(r.cspca)}/{int(r.n)})', va='center', ha='left', fontsize=6.5, color=INK)
        pv = hol.loc[k, 'p_holm'] if row.startswith('OFF-TARGET') else r.p
        ax.text(i, -0.14, fmt_p(pv), ha='center', va='top', fontsize=7, color=INK, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(3))
    ax.set_xticklabels([NAME[k] for k in KEYS], fontsize=7.5)
    ax.set_xlim(-0.5, 2.75)
    top = max(d.hi.max(), d.value.max())
    ax.set_ylim(0, top * 1.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: pct(v)))
    ax.set_ylabel('csPCa-positive cores near a candidate')
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, loc='left', fontsize=8, color=INK)
hs = [plt.Line2D([], [], marker='o', ls='', color=INK2, ms=6), plt.Rectangle((0, 0), 1, 1, color='#d9d8d2'), plt.Line2D([], [], color=INK2, lw=1.0)]
fig.legend(hs, ['Observed', 'Expected by chance: 95% range', 'Expected by chance: mean'], loc='lower center', ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.10))
fig.tight_layout(w_pad=2.0)
save(fig, 'Figure4_offtarget_cores')

# ---------------------------------------------------------------- Figure S1: sensitivity of the comparison of cores away from the targets with chance
rg = pd.read_csv(O + 'A2_results_regress_qc.csv')
TOLS = [(5.0, 'OFF-TARGET, 5 mm'), (7.5, 'OFF-TARGET, 7.5 mm'), (10.0, 'OFF-TARGET, 10 mm (main)'), (12.5, 'OFF-TARGET, 12.5 mm'), (15.0, 'OFF-TARGET, 15 mm')]
VARS = [('regress_qc', 'Top-3\n(main)'), ('top1_qc', 'Top-1'), ('top5_qc', 'Top-5'), ('off15_qc', 'Off-target\n>15 mm')]
have_vars = all(os.path.exists(O + f'A2_results_{t}.csv') for t, _ in VARS)
fig, axs = plt.subplots(2 if have_vars else 1, 3, figsize=(7.2, 5.4 if have_vars else 2.7), squeeze=False)
for i, k in enumerate(KEYS):
    ax = axs[0, i]
    d = rg[rg.model == k].set_index('analysis')
    xs = [t for t, _ in TOLS]
    lo = [d.loc[a, 'lo'] for _, a in TOLS]
    hi = [d.loc[a, 'hi'] for _, a in TOLS]
    ax.fill_between(xs, lo, hi, color='#d9d8d2', lw=0, zorder=1)
    ax.plot(xs, [d.loc[a, 'null_mean'] for _, a in TOLS], color=INK2, lw=1.0, ls='--', zorder=2)
    ax.plot(xs, [d.loc[a, 'value'] for _, a in TOLS], '-o', color=COL[k], lw=2, ms=5, mec='white', mew=1, zorder=3)
    for t, a in TOLS:
        ax.text(t, 0.099, fmt_p(d.loc[a, 'p']).replace('P ', '').replace('= ', ''), ha='center', fontsize=5.8, color=INK2)
    ax.set_xticks(xs)
    ax.set_xlim(3.6, 16.4)
    ax.set_xticklabels(['5', '7.5', '10', '12.5', '15'])
    ax.set_xlabel('Candidate-to-core tolerance (mm)')
    ax.set_ylim(0, 0.108)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: pct(v)))
    ax.set_title(('a  ' if i == 0 else '') + NAME[k], fontsize=8, color=COL[k] if False else INK, loc='left')
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
axs[0, 0].set_ylabel('csPCa-positive cores away from\ntargets near a candidate')
if have_vars:
    for i, k in enumerate(KEYS):
        ax = axs[1, i]
        for j, (t, lab) in enumerate(VARS):
            r = pd.read_csv(O + f'A2_results_{t}.csv')
            r = r[(r.model == k) & (r.analysis == 'OFF-TARGET, 10 mm (main)')].iloc[0]
            ax.add_patch(plt.Rectangle((j - 0.2, r.lo), 0.4, r.hi - r.lo, color='#d9d8d2', lw=0, zorder=1))
            ax.plot([j - 0.2, j + 0.2], [r.null_mean, r.null_mean], color=INK2, lw=1.0, zorder=2)
            ax.plot(j, r.value, 'o', ms=6, color=COL[k], mec='white', mew=1, zorder=3)
            ax.text(j, 0.099, fmt_p(r.p).replace('P ', '').replace('= ', ''), ha='center', fontsize=5.8, color=INK2)
        ax.set_xticks(range(len(VARS)))
        ax.set_xticklabels([v[1] for v in VARS], fontsize=6.3)
        ax.set_xlim(-0.6, len(VARS) - 0.4)
        ax.set_ylim(0, 0.108)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: pct(v)))
        ax.set_title(('b  ' if i == 0 else '') + NAME[k], fontsize=8, color=INK, loc='left')
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axs[1, 0].set_ylabel('csPCa-positive cores away from\ntargets near a candidate')
hs = [plt.Line2D([], [], marker='o', color=INK2, lw=2, ms=5), plt.Rectangle((0, 0), 1, 1, color='#d9d8d2'), plt.Line2D([], [], color=INK2, lw=1.0, ls='--')]
fig.legend(hs, ['Observed (color = model)', 'Expected by chance: 95% range', 'Expected by chance: mean'], loc='lower center', ncol=3, fontsize=7,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(h_pad=1.5, w_pad=1.0)
save(fig, 'FigureS1_offtarget_sensitivity')

# ---------------------------------------------------------------- Table S1: excluded examinations (rendered by D8_submission_package_RIC.py)
e1 = pd.read_csv(O + 'EXCLUDED_SESSIONS.csv')                 # the 14 examinations excluded by the header rule
e2 = pd.read_csv(O + 'EXCLUDED_SESSIONS_v2.csv')              # all 43 excluded examinations
qc = pd.read_csv(O + 'B7_input_content_qc.csv').set_index('Session_ID')
REASON = {'b < 1000': 'b value below 1000 sec/mm²', 'b-value missing in header': 'No b value in the header',
          'high-b image constant or saturated in the gland': 'High-b-value image constant or saturated in the gland', 'ADC constant in the gland': 'ADC constant in the gland'}


def reason(r):
    if r.reason == 'diffusion does not cover >=20% of the gland':
        cov = 100 * (1 - max(qc.loc[r.Session_ID, 'adc_zero_frac'], qc.loc[r.Session_ID, 'hbv_zero_frac']))
        return f'Zero ADC or high-b-value signal in at least 20% of gland voxels (gland coverage, {cov:.1f}%)'
    return REASON[r.reason]


hdr = e2.Session_ID.isin(e1.Session_ID)
s1 = pd.DataFrame({'Examination': e2.Session_ID, 'Rule': np.where(hdr, 'Header (b value)', 'Image content'), 'Reason': [reason(r) for r in e2.itertuples()],
                   'Diffusion series description': e2.dwi_desc.fillna(''), 'b value (sec/mm²)': e2.b_value.map(lambda v: '' if pd.isna(v) else f'{v:.0f}'),
                   '_order': (~hdr).astype(int)})
s1 = s1.sort_values(['_order', 'Examination']).drop(columns='_order')
assert (s1.Rule == 'Header (b value)').sum() == 14 and (s1.Rule == 'Image content').sum() == 29
s1.to_csv(TAB + 'TableS1_exclusions.csv', index=False)
print('table TableS1_exclusions', s1.shape)
print('done')
