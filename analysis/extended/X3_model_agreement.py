# X3: agreement between the three models and its relation to cancer.
#  (a) evaluable cancers: which models localise each cancer (top-3, 10 mm); pairwise Cohen's kappa; cross-tab with the radiologist hit
#  (b) spatial agreement between models: top-1 (and any top-3) candidate centres of two models within 10 mm
#  (c) radiologist targets: number of models agreeing (0-3) vs csPCa at the target; GEE OR per additional agreeing model
#      (adjusted for suspicion score and log PSA density); pairwise kappa and Fleiss kappa of agreement
#  (d) off-target systematic cores: csPCa rate by the number of models with a top-3 candidate within 10 mm (consensus off-target)
#   python analysis/extended/X3_model_agreement.py > outputs/extended/X3.txt
import sys, itertools
sys.path.insert(0, 'analysis/extended')
from xcommon import *
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

T, P = exam_table(), perm_table()
ev = T[T.evaluable == 1].set_index('Session_ID')
L = pd.DataFrame({NAME[m]: (P[(P.model == m) & (P.k == 3)].set_index('Session_ID').loc[ev.index, 'dist'] <= 10).astype(int) for m in MODELS})
L['n_models'] = L.sum(1)
L['rad_hit'] = (ev.rad_dist <= 10).astype(int).to_numpy()
print('(a) evaluable cancers by number of models localising them (top-3, 10 mm), split by radiologist hit (descriptive; no union test)')
print(pd.crosstab(L.n_models, L.rad_hit.map({1: 'radiologist hit', 0: 'radiologist missed'}), margins=True).to_string())
for a, b in itertools.combinations([NAME[m] for m in MODELS], 2):
    print(f'  kappa {a} vs {b}: {cohen_kappa_score(L[a], L[b]):.2f} | both localise {int(((L[a] == 1) & (L[b] == 1)).sum())}, only {a} {int(((L[a] == 1) & (L[b] == 0)).sum())}, only {b} {int(((L[a] == 0) & (L[b] == 1)).sum())}')

# (b) spatial agreement between the models' candidates (all examinations)
cand = candidates(3)
XYZ = ['CoM_LPS_X_mm', 'CoM_LPS_Y_mm', 'CoM_LPS_Z_mm']
C = {m: {s: g.sort_values('Candidate_Rank')[XYZ].to_numpy(float) for s, g in cand[cand.Model == m].groupby('Session_ID')} for m in MODELS}
print('\n(b) spatial agreement between models ')
for a, b in itertools.combinations(MODELS, 2):
    both = [s for s in C[a] if s in C[b]]
    top1 = np.mean([np.linalg.norm(C[a][s][0] - C[b][s][0]) <= 10 for s in both])
    any3 = np.mean([(np.sqrt(((C[a][s][:, None] - C[b][s][None]) ** 2).sum(2)) <= 10).any() for s in both])
    print(f'  {NAME[a]} vs {NAME[b]}: exams with candidates from both {len(both)} | top-1 centres within 10 mm {100 * top1:.1f}% | any top-3 pair within 10 mm {100 * any3:.1f}%')

# (c) radiologist targets: number of agreeing models
tg = pd.read_csv(O + 'B6_targets_final_qc.csv')
tg = tg[tg.path != 'unsampled'].copy()
tg['y'] = (tg.path == 'csPCa').astype(int)
ag = [f'agree_{m}' for m in MODELS]
tg['n_agree'] = tg[ag].sum(1)
print(f'\n(c) {len(tg)} sampled radiologist targets: csPCa rate by number of models agreeing')
for k in range(4):
    q = tg[tg.n_agree == k]
    lo, hi = wilson(int(q.y.sum()), len(q))
    print(f'  {k} model(s): {pct(int(q.y.sum()), len(q))} [Wilson {100 * lo:.1f}-{100 * hi:.1f}%]')
for sc in (3, 4, 5):
    q = tg[tg.score == sc]
    print(f'  score {sc}: ' + ' | '.join(f'{k} models {pct(int(q[q.n_agree == k].y.sum()), int((q.n_agree == k).sum()))}' for k in range(4)))
d = tg[tg.score.isin([3, 4, 5]) & np.isfinite(tg.lpsad)].copy()
d['s4'], d['s5'] = (d.score == 4).astype(int), (d.score == 5).astype(int)
orr = gee_or(d, 'y', ['n_agree', 's4', 's5', 'lpsad'], 'Patient_ID')
print(f'  GEE (n = {len(d)} targets with score 3-5 and PSA density): OR per additional agreeing model {orr[0]:.2f} [{orr[1]:.2f}-{orr[2]:.2f}] {fmt_p(orr[3])}')
d['consensus'] = (d.n_agree >= 2).astype(int)
orc = gee_or(d, 'y', ['consensus', 's4', 's5', 'lpsad'], 'Patient_ID')
print(f'  GEE: at least 2 of 3 models agree vs fewer: OR {orc[0]:.2f} [{orc[1]:.2f}-{orc[2]:.2f}] {fmt_p(orc[3])}')
d3 = d[d.score == 3]
or3 = gee_or(d3, 'y', ['n_agree', 'lpsad'], 'Patient_ID')
print(f'  score 3 only (n = {len(d3)}): OR per additional agreeing model {or3[0]:.2f} [{or3[1]:.2f}-{or3[2]:.2f}] {fmt_p(or3[3])}')
for a, b in itertools.combinations(MODELS, 2):
    print(f'  kappa of target agreement {NAME[a]} vs {NAME[b]}: {cohen_kappa_score(tg[f"agree_{a}"], tg[f"agree_{b}"]):.2f}')
print(f'  Fleiss kappa (3 models): {fleiss_kappa(aggregate_raters(tg[ag].to_numpy())[0]):.2f}')

# (d) consensus off-target
ms = master()
cores = cores_757()
EX = offtarget_cores(cores, ms)
C3 = {m: {s: g[XYZ].to_numpy(float) for s, g in cand[cand.Model == m].groupby('Session_ID')} for m in MODELS}
rows = []
for e in EX:
    s = e['Session_ID']
    near_m = np.zeros((len(MODELS), len(e['y'])), bool)
    for i, m in enumerate(MODELS):
        if s in C3[m]:
            near_m[i] = p2s(C3[m][s], e['A'], e['B']).min(0) <= 10
    for j in np.where(~e['near'])[0]:
        rows.append(dict(Session_ID=s, y=e['y'][j], n_models=int(near_m[:, j].sum())))
Q = pd.DataFrame(rows)
print(f'\n(d) off-target systematic cores: {len(Q)} cores, {int(Q.y.sum())} csPCa')
for k in range(4):
    q = Q[Q.n_models == k]
    lo, hi = wilson(int(q.y.sum()), len(q))
    print(f'  near {k} model(s): {pct(int(q.y.sum()), len(q))} [Wilson {100 * lo:.1f}-{100 * hi:.1f}%; cores are clustered, interval is approximate]')
Q['ge2'] = (Q.n_models >= 2).astype(int)
Q = Q.merge(T[['Session_ID', 'Patient_ID']], on='Session_ID')
o2 = gee_or(Q, 'y', ['ge2'], 'Patient_ID')
print(f'  GEE (clustered by patient): near >= 2 models vs fewer: OR {o2[0]:.2f} [{o2[1]:.2f}-{o2[2]:.2f}] {fmt_p(o2[3])} (no chance model; descriptive)')
L.to_csv(R + 'X3a_localisation_by_model.csv')
Q.to_csv(R + 'X3d_offtarget_consensus.csv', index=False)
