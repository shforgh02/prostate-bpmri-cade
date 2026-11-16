# X4: clinical added value and biopsy decision.
#  (a) examination level, suspicion score 3-5 with PSA density and age: cross-fitted logistic models (5 folds, as in A3)
#      score; + log PSA density; + age; + each model (logit of the highest candidate probability); + all three models;
#      + "model agrees with a radiologist target". AUROC with bootstrap CI of the gain over score + PSA density.
#  (b) decision curve analysis (net benefit, net biopsies avoided per 100 men) at risk thresholds 5-40%.
#  (c) simple biopsy-deferral rules on all examinations: biopsies avoided and csPCa / grade group >= 3 missed.
#  (d) systematic-only cancers split by whether the radiologist target was near the positive core (completes X2).
#  (e) model alone, AUROC within suspicion-score and PSA-density strata.
#   python analysis/extended/X4_clinical_value.py > outputs/extended/X4.txt
import sys
sys.path.insert(0, 'analysis/extended')
from xcommon import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T, P = exam_table(), perm_table()
tg = pd.read_csv(O + 'B6_targets_final_qc.csv')
for m in MODELS:
    T[f'agree_{m}'] = T.Session_ID.map(tg.groupby('Session_ID')[f'agree_{m}'].max()).fillna(0).astype(int)
    T[f'lp_{m}'] = np.log(np.clip(T[f'maxprob_{m}'], 1e-4, 1 - 1e-4) / (1 - np.clip(T[f'maxprob_{m}'], 1e-4, 1 - 1e-4)))
T['n_agree'] = T[[f'agree_{m}' for m in MODELS]].sum(1)
T['lpsad'] = np.log(T.psad)
T['s4'], T['s5'] = (T.score == 4).astype(int), (T.score == 5).astype(int)
D = T[T.score.isin([3, 4, 5]) & np.isfinite(T.lpsad) & T.age.notna()].reset_index(drop=True)
y = D.cspca.to_numpy(int)
rng = np.random.default_rng(1)


def cross_fit(cols):
    X = D[cols].to_numpy(float)
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        p[te] = LogisticRegression(C=1e6, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


BASE = ['s4', 's5', 'lpsad']
SPEC = {'suspicion score': ['s4', 's5'], 'PSA density': ['lpsad'], 'age': ['age'], 'score + PSA density': BASE, 'score + PSA density + age': BASE + ['age']}
for m in MODELS:
    SPEC[f'{NAME[m]} alone'] = [f'lp_{m}']
    SPEC[f'score + PSA density + {NAME[m]}'] = BASE + [f'lp_{m}']
    SPEC[f'score + PSA density + {NAME[m]} agrees with target'] = BASE + [f'agree_{m}']
SPEC['score + PSA density + all three models'] = BASE + [f'lp_{m}' for m in MODELS]
SPEC['score + PSA density + number of models agreeing'] = BASE + ['n_agree']
PR = {k: cross_fit(v) for k, v in SPEC.items()}
ref = PR['score + PSA density']
print(f'(a) EXAMINATION LEVEL: n = {len(D)} (score 3-5 with PSA density and age), csPCa {y.sum()} | cross-fitted AUROC; gain vs score + PSA density [bootstrap 95% CI]')
rows = []
for k, p in PR.items():
    a = roc_auc_score(y, p)
    g = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            g.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], ref[i]))
    lo, hi = np.percentile(g, [2.5, 97.5])
    rows.append(dict(model=k, auroc=a, gain=a - roc_auc_score(y, ref), lo=lo, hi=hi))
    print(f'  {k:52s} AUROC {a:.3f} | gain {a - roc_auc_score(y, ref):+.3f} [{lo:+.3f}, {hi:+.3f}]')
pd.DataFrame(rows).to_csv(R + 'X4a_exam_level.csv', index=False)

# (b) decision curves
TH = np.arange(0.05, 0.41, 0.05)
def nb(p, t):
    tp, fp = ((p >= t) & (y == 1)).sum(), ((p >= t) & (y == 0)).sum()
    return tp / len(y) - fp / len(y) * t / (1 - t)
nb_all = {t: y.mean() - (1 - y.mean()) * t / (1 - t) for t in TH}
print('\n(b) DECISION CURVES: net benefit (net biopsies avoided per 100 men vs biopsy-all), cross-fitted probabilities')
SHOW = ['score + PSA density', 'score + PSA density + age'] + [f'score + PSA density + {NAME[m]}' for m in MODELS] + ['score + PSA density + all three models', 'score + PSA density + number of models agreeing']
print('  threshold   ' + ' | '.join(f'{t:.2f}' for t in TH))
print('  biopsy all  ' + ' | '.join(f'{nb_all[t]:.3f}' for t in TH))
dca = []
for k in SHOW:
    vals = [nb(PR[k], t) for t in TH]
    avoid = [100 * (v - nb_all[t]) / (t / (1 - t)) for v, t in zip(vals, TH)]
    dca += [dict(model=k, threshold=t, net_benefit=v, net_biopsies_avoided_per_100=a) for t, v, a in zip(TH, vals, avoid)]
    print(f'  {k[:46]:46s} ' + ' | '.join(f'{v:.3f} ({a:+.0f})' for v, a in zip(vals, avoid)))
pd.DataFrame(dca).to_csv(R + 'X4b_decision_curves.csv', index=False)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(TH, [nb_all[t] for t in TH], color='#888780', lw=1.5, label='Biopsy all')
ax.axhline(0, color='#c3c2b7', lw=1, label='Biopsy none')
for k, c in zip(SHOW[:5], ['#1f1f1e', '#5f5e5a', '#2a78d6', '#eb6834', '#1baf7a']):
    ax.plot(TH, [nb(PR[k], t) for t in TH], color=c, lw=1.8, label=k.replace('score + PSA density', 'Score + PSAD'))
ax.set_xlabel('Threshold probability of csPCa'); ax.set_ylabel('Net benefit'); ax.legend(fontsize=7, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(R + 'X4b_decision_curves.png', dpi=200); plt.close(fig)

# (c) biopsy-deferral rules on all examinations with a score and PSA density
A = T[T.score.isin([1, 2, 3, 4, 5]) & np.isfinite(T.psad)].copy()
ncs, nhg = int(A.cspca.sum()), int((A.grade >= 3).sum())
print(f'\n(c) DEFERRAL RULES (retrospective; all {len(A)} examinations with a score and PSA density; csPCa {ncs}, grade group >= 3 {nhg})')
R0 = A.score <= 2
R1 = (A.score <= 3) & (A.psad < 0.15)
rules = [('score 1-2', R0), ('score <= 3 and PSAD < 0.15', R1)]
for m in MODELS:
    rules.append((f'score <= 3 and PSAD < 0.15 and {NAME[m]} does not agree with a target', R1 & (A[f'agree_{m}'] == 0)))
rules += [('score <= 3 and PSAD < 0.15 and no model agrees', R1 & (A.n_agree == 0)), ('score <= 3 and PSAD < 0.15 and fewer than 3 models agree', R1 & (A.n_agree < 3)),
          ('score 3 and no model agrees (any PSAD)', (A.score == 3) & (A.n_agree == 0))]
rr = []
for lab, msk in rules:
    q = A[msk]
    mi, hg = int(q.cspca.sum()), int((q.grade >= 3).sum())
    lo, hi = wilson(mi, len(q))
    rr.append(dict(rule=lab, deferred=len(q), csPCa_missed=mi, gg3_missed=hg))
    print(f'  {lab:66s} deferred {len(q):3d} ({100 * len(q) / len(A):4.1f}%) | csPCa missed {mi:3d} ({100 * mi / ncs:4.1f}% of csPCa; '
          f'{100 * mi / max(len(q), 1):4.1f}% of deferred [{100 * lo:.1f}-{100 * hi:.1f}%]) | GG>=3 missed {hg}')
pd.DataFrame(rr).to_csv(R + 'X4c_deferral_rules.csv', index=False)

# (d) systematic-only cancers split by the radiologist target
ev = T[T.evaluable == 1].set_index('Session_ID')
print('\n(d) SYSTEMATIC-ONLY CANCERS (targeted cores negative or absent, C = 0), split by radiologist target position')
for lab, msk in (('radiologist target within 10 mm (targeting missed)', (ev.C == 0) & (ev.rad_dist <= 10)),
                 ('radiologist target > 10 mm or absent', (ev.C == 0) & ~(ev.rad_dist <= 10))):
    ids = msk[msk].index
    cells = []
    for m in MODELS:
        d = P[(P.model == m) & (P.k == 3)].set_index('Session_ID').loc[ids]
        o = int((d.dist <= 10).sum()); e, p = pbin_tail(o, d['perm_10.0'].to_numpy())
        cells.append(f'{NAME[m]} {o}/{len(ids)} (chance {e:.1f}) {fmt_p(p)}')
    print(f'  {lab:52s} n={len(ids):2d} | ' + ' | '.join(cells))

# (e) model alone within strata
print('\n(e) MODEL ALONE: AUROC within strata (all examinations with the variable)')
for lab, msk in (('score 3', T.score == 3), ('score 4', T.score == 4), ('score 5', T.score == 5), ('PSAD < 0.15', T.psad < 0.15), ('PSAD >= 0.15', T.psad >= 0.15)):
    q = T[msk]
    print(f'  {lab:14s} n={len(q):3d} csPCa {int(q.cspca.sum()):3d} | ' + ' | '.join(f'{NAME[m]} {roc_auc_score(q.cspca, q[f"maxprob_{m}"]):.3f}' for m in MODELS)
          + f' | PSAD {roc_auc_score(q.cspca[q.psad.notna()], q.psad.dropna()):.3f}')
