# X8:
#  (a) target-level ("lesion-level") view of agreement: sensitivity, specificity, PPV and NPV of a model candidate at the radiologist target
#      for csPCa at that target; per model, at least 2 models, all 3; overall and by suspicion score
#  (b) high-grade outcome (grade group >= 3) at the examination level: cross-fitted AUROC of score + PSA density (+ age) (+ each model)
#  (c) distance definitions: top-3 localisation by candidate centre, candidate peak and (SegResNet only) candidate surface, vs radiologist
#   python analysis/extended/X8_target_level_and_highgrade.py > outputs/extended/X8.txt
import sys
sys.path.insert(0, 'analysis/extended')
from xcommon import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

T = exam_table()
tg = pd.read_csv(O + 'B6_targets_final_qc.csv')
tg = tg[tg.path != 'unsampled'].copy()
tg['y'] = (tg.path == 'csPCa').astype(int)
tg['n_agree'] = tg[[f'agree_{m}' for m in MODELS]].sum(1)
print(f'(a) TARGET LEVEL: {len(tg)} sampled radiologist targets, {int(tg.y.sum())} with csPCa (targets are clustered within men; descriptive)')


def metrics(flag, y):
    tp, fp = int(((flag == 1) & (y == 1)).sum()), int(((flag == 1) & (y == 0)).sum())
    fn, tn = int(((flag == 0) & (y == 1)).sum()), int(((flag == 0) & (y == 0)).sum())
    return f'sens {pct(tp, tp + fn)} | spec {pct(tn, tn + fp)} | PPV {pct(tp, tp + fp)} | NPV {pct(tn, tn + fn)}'


rows = []
for sub_lab, sub in (('all scores', tg), ('score 3', tg[tg.score == 3]), ('score 4', tg[tg.score == 4]), ('score 5', tg[tg.score == 5])):
    print(f'-- {sub_lab} (n = {len(sub)})')
    for lab, flag in [(NAME[m], sub[f'agree_{m}']) for m in MODELS] + [('at least 2 models', (sub.n_agree >= 2).astype(int)), ('all 3 models', (sub.n_agree == 3).astype(int))]:
        print(f'   {lab:18s} {metrics(flag, sub.y)}')
        rows.append(dict(subset=sub_lab, rule=lab, n=len(sub), positive=int(flag.sum()), csPCa=int(sub.y.sum()), tp=int(((flag == 1) & (sub.y == 1)).sum())))
pd.DataFrame(rows).to_csv(R + 'X8a_target_level.csv', index=False)

# (b) high-grade outcome
for m in MODELS:
    T[f'lp_{m}'] = np.log(np.clip(T[f'maxprob_{m}'], 1e-4, 1 - 1e-4) / (1 - np.clip(T[f'maxprob_{m}'], 1e-4, 1 - 1e-4)))
T['lpsad'] = np.log(T.psad)
T['s4'], T['s5'] = (T.score == 4).astype(int), (T.score == 5).astype(int)
D = T[T.score.isin([3, 4, 5]) & np.isfinite(T.lpsad) & T.age.notna()].reset_index(drop=True)
y = (D.grade >= 3).astype(int).to_numpy()
rng = np.random.default_rng(2)


def cf(cols):
    X = D[cols].to_numpy(float); p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        p[te] = LogisticRegression(C=1e6, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


base = cf(['s4', 's5', 'lpsad'])
print(f'\n(b) HIGH-GRADE OUTCOME (grade group >= 3): n = {len(D)}, events {y.sum()} | cross-fitted AUROC; gain vs score + PSA density [bootstrap 95% CI]')
for lab, cols in [('score + PSA density', ['s4', 's5', 'lpsad']), ('score + PSA density + age', ['s4', 's5', 'lpsad', 'age'])] + \
                 [(f'{NAME[m]} alone', [f'lp_{m}']) for m in MODELS] + [(f'score + PSA density + {NAME[m]}', ['s4', 's5', 'lpsad', f'lp_{m}']) for m in MODELS]:
    p = cf(cols)
    g = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            g.append(roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], base[i]))
    lo, hi = np.percentile(g, [2.5, 97.5])
    print(f'  {lab:40s} AUROC {roc_auc_score(y, p):.3f} | gain {roc_auc_score(y, p) - roc_auc_score(y, base):+.3f} [{lo:+.3f}, {hi:+.3f}]')

# (c) distance definitions
c = candidates(3)
g = c.groupby(['Model', 'Session_ID'])[['CoM_Dist_csPCa_mm', 'Peak_Dist_csPCa_mm', 'Surf3D_Dist_csPCa_mm']].min()
ev = T[T.evaluable == 1].set_index('Session_ID')
n = len(ev)
print(f'\n(c) DISTANCE DEFINITIONS, top-3 candidates, {n} evaluable cancers (observed only; no chance model)')
print('   radiologist: centre ' + ' | '.join(f'{t:g}mm {100 * (ev.rad_dist <= t).mean():.1f}%' for t in (0, 2.5, 5, 10)) +
      ' ; surface ' + ' | '.join(f'{t:g}mm {100 * (ev.rad_surf_dist <= t).mean():.1f}%' for t in (0, 2.5, 5, 10)) + f' ; needle through target {100 * (ev.rad_puncture == 1).mean():.1f}%')
for m in MODELS:
    q = g.loc[m].reindex(ev.index) if m in g.index.get_level_values(0) else None
    parts = []
    for col, lab in (('CoM_Dist_csPCa_mm', 'centre'), ('Peak_Dist_csPCa_mm', 'peak'), ('Surf3D_Dist_csPCa_mm', 'surface')):
        if q is None or q[col].notna().sum() == 0:
            parts.append(f'{lab}: not available'); continue
        v = q[col].fillna(np.inf)
        parts.append(f'{lab} ' + ' | '.join(f'{t:g}mm {100 * (v <= t).mean():.1f}%' for t in (0, 2.5, 5, 10)))
    print(f'   {NAME[m]:9s} ' + ' ; '.join(parts))
print('   (surface distance and needle-through-candidate are available only for SegResNet, whose candidate masks were stored in earlier work)')
