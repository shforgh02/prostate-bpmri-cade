# Systematic cores away from radiologist targets compared with chance.
# Off-target systematic cores = valid-trajectory systematic cores >10 mm from every targeted core and from the radiologist target centre.
# Test: csPCa rate among off-target cores within 10 mm of a top-3 candidate, against the between-patient permutation null
#       (candidate sets exchanged between exams with the same number of candidates, mapped through the prostate bounding box).
# Also: within-patient AUROC of distance-to-nearest-candidate, bootstrap over patients. Works for any number of models.
# Run from the repository root:
#   python analysis/A2_core_level_off_target.py [--exclude csv] [--candidates csv] [--tag name] [--nperm 2000]
import sys, os, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SD, OUT, CC, args, load, p2s, cand_xyz, bbox

a = args()
ep, cand, cores, ms, models = load(a)
rng = np.random.default_rng(7)
cv = cores[cores.Has_Valid_Physical_Trajectory == 1]
EX = {}
for s, g in cv.groupby('Session_ID_Matched'):
    sy = g[g.Is_Systematic_Core == 1]
    tg = g[g.Is_Targeted_Core == 1]
    if len(sy) == 0:
        continue
    A, B = sy[CC[:3]].to_numpy(float), sy[CC[3:]].to_numpy(float)
    near = np.zeros(len(sy), bool)
    if len(tg):   # targeted cores (sampled along their length) mark where the radiologist target is
        TA, TB = tg[CC[:3]].to_numpy(float), tg[CC[3:]].to_numpy(float)
        near |= p2s(np.concatenate([TA + (TB - TA) * f for f in np.linspace(0, 1, 9)]), A, B).min(0) <= a.offdist
    h = ms.loc[s, ['Human_Rad_CoM_LPS_X_mm', 'Human_Rad_CoM_LPS_Y_mm', 'Human_Rad_CoM_LPS_Z_mm']].to_numpy(float)
    if np.isfinite(h).all():
        near |= p2s(h[None], A, B)[0] <= a.offdist
    mn, sp = bbox(ms, s)
    EX[s] = dict(A=A, B=B, y=sy.Is_csPCa.to_numpy(int), near_rad=near, mn=mn, sp=sp)
S = list(EX)
y = np.concatenate([EX[s]['y'] for s in S])
nr = np.concatenate([EX[s]['near_rad'] for s in S])
sid = np.concatenate([[i] * len(EX[s]['y']) for i, s in enumerate(S)])
cs_exam = ep.set_index('Session_ID').Is_csPCa
in_cs = np.isin(sid, [i for i, s in enumerate(S) if cs_exam.get(s, 0) == 1])
print(f'cohort {len(ep)} exams | systematic cores {len(y)} in {len(S)} exams | near target {nr.sum()} (csPCa {y[nr].mean():.1%}) | '
      f'OFF-TARGET {(~nr).sum()} (csPCa+ {y[~nr].sum()}, {y[~nr].mean():.1%}) | models {models}')


def rate(d, m, tol):
    k = m & (d <= tol)
    return y[k].sum() / max(k.sum(), 1), int(y[k].sum()), int(k.sum())


def auc(yy, sc):
    """AUROC by the rank-sum formula (ties get average ranks)."""
    r = pd.Series(sc).rank().to_numpy()
    n1 = yy.sum()
    n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# patients with at least one positive and one negative off-target core
ELIG = [np.where((sid == i) & ~nr)[0] for i in range(len(S))]
ELIG = [k for k in ELIG if 0 < y[k].sum() < len(k)]


ANALYSES = (('OFF-TARGET, 10 mm (main)', ~nr, 10), ('OFF-TARGET, 5 mm', ~nr, 5), ('OFF-TARGET, 7.5 mm', ~nr, 7.5), ('OFF-TARGET, 12.5 mm', ~nr, 12.5),
            ('OFF-TARGET, 15 mm', ~nr, 15), ('near radiologist target, 10 mm (positive control)', nr, 10))
out = []
for model in models:
    C = {s: v for s, v in cand_xyz(cand, model).items() if s in EX}
    NORM = {s: (C[s] - EX[s]['mn']) / EX[s]['sp'] for s in C}
    byk = {}
    for s, v in NORM.items():
        byk.setdefault(len(v), []).append(s)

    def dist(assign):
        parts = []
        for s in S:
            e = EX[s]
            if s not in C:
                parts.append(np.full(len(e['y']), 99.0))
                continue
            P = C[s] if assign is None else e['mn'] + NORM[assign[s]] * e['sp']
            parts.append(p2s(P, e['A'], e['B']).min(0))
        return np.concatenate(parts)

    d0 = dist(None)
    PERM, WP = [], []
    for b in range(a.nperm):
        assign = {}
        for k, lst in byk.items():
            for x, z in zip(lst, rng.permutation(lst)):
                assign[x] = z
        d = dist(assign)
        PERM.append([rate(d, m, tol)[0] for _, m, tol in ANALYSES])
        WP.append(np.mean([auc(y[k], -d[k]) for k in ELIG]))
    PERM = np.array(PERM)
    WP = np.array(WP)
    print(f'\n[{model}]')
    for j, (lab, m, tol) in enumerate(ANALYSES):
        r, npos, n = rate(d0, m, tol)
        rest = y[m & (d0 > tol)].mean()
        p = (np.sum(PERM[:, j] >= r) + 1) / (a.nperm + 1)
        lo, hi = np.percentile(PERM[:, j], [2.5, 97.5])
        print(f'  {lab:50s} cores near candidate {n:5d}, csPCa+ {npos:4d} = {r:5.1%} | not near {rest:5.1%} | permutation null {PERM[:, j].mean():5.1%} [{lo:.1%}-{hi:.1%}] p={p:.4f}')
        out.append(dict(model=model, analysis=lab, n=n, cspca=npos, value=r, rate_not_near=rest, null_mean=PERM[:, j].mean(), lo=lo, hi=hi, p=p))
    W = np.array([auc(y[k], -d0[k]) for k in ELIG])
    bs = [rng.choice(W, len(W)).mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pw = (np.sum(WP >= W.mean()) + 1) / (a.nperm + 1)
    kk = ~nr & in_cs
    pooled = roc_auc_score(y[kk], -d0[kk])
    print(f'  within-patient off-target AUROC {W.mean():.3f} [bootstrap {lo:.3f}-{hi:.3f}] over {len(W)} patients | permutation null {WP.mean():.3f} '
          f'[{np.percentile(WP, 2.5):.3f}-{np.percentile(WP, 97.5):.3f}] p={pw:.4f} | pooled off-target AUROC in csPCa patients {pooled:.3f}')
    out.append(dict(model=model, analysis='within-patient off-target AUROC', n=len(W), value=W.mean(), null_mean=WP.mean(), lo=lo, hi=hi, p=pw))
    out.append(dict(model=model, analysis='pooled off-target AUROC, csPCa patients', n=int(kk.sum()), value=pooled, null_mean=0.5))
pd.DataFrame(out).to_csv(OUT + f'A2_results_{a.tag}.csv', index=False)
