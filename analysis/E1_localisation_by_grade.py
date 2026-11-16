# Localisation of evaluable cancers by tumour grade, by the number of csPCa-positive cores and by suspicion score, for each model,
# against the same between-patient permutation chance model as A1 (10-mm tolerance, top-3 candidates). Also reports the radiologist hit rate.
#   python analysis/E1_localisation_by_grade.py --exclude outputs/EXCLUDED_SESSIONS_v2.csv \
#          --candidates outputs/CANDIDATES_3MODELS.csv --tag final_qc
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SD, OUT, CC, args, load, p2s, pbin_tail, cand_xyz, bbox

a = args()
ep, cand, cores, ms, models = load(a)
D = ep[(ep.Is_csPCa == 1) & (ep.Has_Valid_3D_Trajectory == 1)].reset_index(drop=True)
pos = cores[cores.Is_Valid_csPCa_Trajectory == 1]
SEG = {s: (g[CC[:3]].to_numpy(float), g[CC[3:]].to_numpy(float)) for s, g in pos.groupby('Session_ID_Matched')}
npos = pos.groupby('Session_ID_Matched').size()
BB = {s: bbox(ms, s) for s in ep.Session_ID}
D['grade'] = ms.loc[D.Session_ID, 'Sess_Max_ISUP'].to_numpy()
D['n_pos_cores'] = npos.reindex(D.Session_ID).fillna(0).to_numpy()
D['rad_hit'] = (D.Human_Target_CoM_Dist_csPCa_mm <= 10).to_numpy()
S = D.Exact_Exam_UCLA_Score
GROUPS = [('all evaluable cancers', np.ones(len(D), bool)),
          ('grade group 2', (D.grade == 2).to_numpy()), ('grade group 3-5', (D.grade >= 3).to_numpy()),
          ('1 positive core', (D.n_pos_cores == 1).to_numpy()), ('2 or more positive cores', (D.n_pos_cores >= 2).to_numpy()),
          ('suspicion score 3', (S == 3).to_numpy()), ('suspicion score 4-5', S.isin([4, 5]).to_numpy())]
out = []
print(f'cohort {len(ep)} exams | evaluable cancers {len(D)} | models {models}')
for g, m in GROUPS:
    out.append(dict(model='radiologist target', subset=g, n=int(m.sum()), observed=int(D.rad_hit[m].sum())))
for model in models:
    C = cand_xyz(cand, model)
    NORM = {s: (C[s] - BB[s][0]) / BB[s][1] for s in C}
    obs, perm = [], []
    for s in D.Session_ID:
        A, B = SEG[s]
        mn, sp = BB[s]
        k = len(C.get(s, []))
        own = p2s(C[s], A, B).min(1) if k else np.array([np.inf])
        donors = [v for j, v in NORM.items() if j != s and len(v) == k] if k else []
        dmin = np.array([p2s(mn + v * sp, A, B).min() for v in donors]) if donors else np.array([np.inf])
        obs.append(int(own.min() <= 10))
        perm.append(float((dmin <= 10).mean()))
    obs, perm = np.array(obs), np.array(perm)
    print(f'\n[{model}]')
    for g, m in GROUPS:
        o = int(obs[m].sum())
        e, p = pbin_tail(o, perm[m])
        print(f'  {g:28s} n={m.sum():3d} | localised {o:3d} ({o / m.sum():5.1%}) | chance {e:6.1f} ({e / m.sum():5.1%}) p={p:.4f} '
              f'| radiologist target hit {int(D.rad_hit[m].sum())} ({D.rad_hit[m].mean():.1%})')
        out.append(dict(model=model, subset=g, n=int(m.sum()), observed=o, perm_expected=round(e, 1), perm_p=p))
pd.DataFrame(out).to_csv(OUT + f'E1_results_{a.tag}.csv', index=False)
