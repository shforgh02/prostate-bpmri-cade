# Examination-level correspondence of top-3 candidates against two chance models (uniform; between-patient permutation),
# overall and split by whether the radiologist target hit, at 10 / 7.5 / 5 / 2.5 mm. Works for any number of models.
# Run from the repository root:
#   python analysis/A1_chance_models_and_tolerance.py [--exclude csv] [--candidates csv] [--tag name]
import sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SD, OUT, CC, args, load, p2s, pbin_tail, cand_xyz, bbox

a = args()
ep, cand, cores, ms, models = load(a)
nul = pd.read_csv(SD + 'PER_EXAM_NULL_OCCUPANCY_AND_ALLOCATION.csv')
D = ep[(ep.Is_csPCa == 1) & (ep.Has_Valid_3D_Trajectory == 1)].merge(nul, on='Session_ID').reset_index(drop=True)
pos = cores[cores.Is_Valid_csPCa_Trajectory == 1]
SEG = {s: (g[CC[:3]].to_numpy(float), g[CC[3:]].to_numpy(float)) for s, g in pos.groupby('Session_ID_Matched')}
BB = {s: bbox(ms, s) for s in ep.Session_ID}
TOLS = (10.0, 7.5, 5.0, 2.5)
out = []
print(f'cohort: {len(ep)} exams, {len(D)} trajectory-evaluable csPCa | models: {models}')
for model in models:
    C = cand_xyz(cand, model)
    NORM = {s: (C[s] - BB[s][0]) / BB[s][1] for s in C}
    rows = []
    for s in D.Session_ID:
        A, B = SEG[s]
        mn, sp = BB[s]
        k = len(C.get(s, []))
        own = p2s(C[s], A, B).min(1) if k else np.array([np.inf])
        # permutation null: candidate sets of other exams with the same number of candidates, mapped into this gland
        donors = [v for j, v in NORM.items() if j != s and len(v) == k] if k else []
        dmin = np.array([p2s(mn + v * sp, A, B).min() for v in donors]) if donors else np.array([np.inf])
        r = {'Session_ID': s, 'k': k}
        for t in TOLS:
            r[f'obs_{t}'] = int(own.min() <= t)
            r[f'perm_{t}'] = float((dmin <= t).mean())
        rows.append(r)
    R = pd.DataFrame(rows).merge(D, on='Session_ID')
    print(f'\n[{model}] top-3 candidates in these exams: {int(R.k.sum())}')
    for t in TOLS:
        Ht = (R.Human_Target_CoM_Dist_csPCa_mm <= t).to_numpy()   # no contour (NaN) counts as miss
        un = 1 - (1 - R[f'q_{t}mm'].to_numpy(float)) ** R.k.to_numpy()
        print(f'  -- tolerance {t} mm; radiologist target hit {Ht.sum()}/{len(R)}')
        subsets = (('all csPCa', np.ones(len(R), bool)), ('rad target HIT', Ht), ('rad target MISS', ~Ht),
                   ('rad MISS & targeted cores neg/absent', (~Ht) & (R.Targeted_Biopsy_csPCa_Yield_C == 0).to_numpy()))
        for nm, m in subsets:
            o = int(R.loc[m, f'obs_{t}'].sum())
            e1, p1 = pbin_tail(o, un[m])
            e2, p2 = pbin_tail(o, R.loc[m, f'perm_{t}'].to_numpy())
            print(f'     {nm:38s} n={m.sum():3d} obs {o:3d} ({o / max(m.sum(), 1):5.1%}) | uniform null {e1:6.1f} p={p1:.4f} | permutation null {e2:6.1f} p={p2:.4f}')
            out.append(dict(model=model, tol_mm=t, subset=nm, n=int(m.sum()), observed=o, uniform_expected=round(e1, 1), uniform_p=p1, perm_expected=round(e2, 1), perm_p=p2))
pd.DataFrame(out).to_csv(OUT + f'A1_results_{a.tag}.csv', index=False)
