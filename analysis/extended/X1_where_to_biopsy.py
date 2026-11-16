# X1: "where to biopsy".
#  (a) Localisation of the evaluable cancers for every model x top-k (1, 2, 3, 5) x tolerance (2.5-15 mm), against the permutation
#      chance model, next to the radiologist's geometric hit rate (target centre and target surface) and the recorded targeted-biopsy yield.
#  (b) Candidate-level yield: for each candidate rank, how often a candidate was sampled (a core within 10 mm) and how often it lay within
#      10 mm of a csPCa core; separately for candidates away from the radiologist target (> 10 mm, "novel").
#   python analysis/extended/X1_where_to_biopsy.py > outputs/extended/X1.txt
import sys
sys.path.insert(0, 'analysis/extended')
from xcommon import *

T, P = exam_table(), perm_table()
ev = T[T.evaluable == 1]
n = len(ev)
rows = []
print(f'(a) LOCALISATION of {n} evaluable cancers ')
print('radiologist target (centre of mass within t mm of a csPCa core) | target surface within t mm | recorded targeted-biopsy yield')
for t in TOLS:
    hc, hs = int((ev.rad_dist <= t).sum()), int((ev.rad_surf_dist <= t).sum())
    rows.append(dict(reader='radiologist', definition='target centre', k='', tol_mm=t, n=n, hit=hc))
    rows.append(dict(reader='radiologist', definition='target surface', k='', tol_mm=t, n=n, hit=hs))
    print(f'  {t:4.1f} mm: centre {pct(hc, n)} | surface {pct(hs, n)}')
pz = int((ev.rad_puncture == 1).sum())
cy = int((ev.C == 1).sum())
rows += [dict(reader='radiologist', definition='needle through target (0 mm)', n=n, hit=pz), dict(reader='radiologist', definition='recorded targeted-biopsy yield (C)', n=n, hit=cy)]
print(f'  needle through the target (0 mm): {pct(pz, n)} | recorded targeted-biopsy yield: {pct(cy, n)}')
for mod in MODELS:
    print(f'\n[{NAME[mod]}]  observed (chance by permutation, p) ; mean candidates per evaluable exam')
    for k in (1, 2, 3, 5):
        d = P[(P.model == mod) & (P.k == k)]
        cells = []
        for t in TOLS:
            o = int((d.dist <= t).sum())
            e, p = pbin_tail(o, d[f'perm_{t}'].to_numpy())
            rows.append(dict(reader=NAME[mod], definition='candidate centre', k=k, tol_mm=t, n=len(d), hit=o, chance=round(e, 1), p=p))
            cells.append(f'{t:g}mm {100 * o / len(d):4.1f}% ({100 * e / len(d):4.1f}%)')
        print(f'  top-{k} [{d.n_cand.mean():.2f} cand]: ' + ' | '.join(cells))
pd.DataFrame(rows).to_csv(R + 'X1a_localisation_grid.csv', index=False)

# (b) candidate-level yield
print('\n(b) CANDIDATE-LEVEL YIELD (all examinations; sampled = any valid core within 10 mm; positive = csPCa core within 10 mm)')
ms = master()
cores = cores_757()
cv = cores[cores.Has_Valid_Physical_Trajectory == 1]
ALL = {s: (g[CC[:3]].to_numpy(float), g[CC[3:]].to_numpy(float), g.Is_csPCa.to_numpy(int)) for s, g in cv.groupby('Session_ID_Matched')}
cand = candidates(5)
out = []
for _, c in cand.iterrows():
    s = c.Session_ID
    if s not in ALL:
        continue
    A, B, y = ALL[s]
    d = p2s(c[['CoM_LPS_X_mm', 'CoM_LPS_Y_mm', 'CoM_LPS_Z_mm']].to_numpy(float)[None], A, B)[0]
    h = ms.loc[s, ['Human_Rad_CoM_LPS_X_mm', 'Human_Rad_CoM_LPS_Y_mm', 'Human_Rad_CoM_LPS_Z_mm']].to_numpy(float)
    drad = float(np.sqrt(((h - c[['CoM_LPS_X_mm', 'CoM_LPS_Y_mm', 'CoM_LPS_Z_mm']].to_numpy(float)) ** 2).sum())) if np.isfinite(h).all() else np.inf
    out.append(dict(model=c.Model, rank=int(c.Candidate_Rank), sampled=int((d <= 10).any()), positive=int((d[y == 1] <= 10).any()) if y.any() else 0,
                    n_cores_10mm=int((d <= 10).sum()), novel=int(drad > 10)))
Y = pd.DataFrame(out)
Y.to_csv(R + 'X1b_candidate_yield.csv', index=False)
for mod in MODELS:
    print(f'[{NAME[mod]}]')
    for lab, sub in (('all candidates', Y[Y.model == mod]), ('novel (> 10 mm from target)', Y[(Y.model == mod) & (Y.novel == 1)]),
                     ('at the target (<= 10 mm)', Y[(Y.model == mod) & (Y.novel == 0)])):
        for r in sorted(sub['rank'].unique()):
            q = sub[sub['rank'] == r]
            smp = q[q.sampled == 1]
            print(f'  {lab:28s} rank {r}: n={len(q):4d} | sampled {100 * len(smp) / len(q):5.1f}% (median {q.n_cores_10mm.median():.0f} cores) '
                  f'| csPCa among sampled {pct(int(smp.positive.sum()), len(smp))}')
