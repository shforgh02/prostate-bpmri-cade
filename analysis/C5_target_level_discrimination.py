# Target-level discrimination of csPCa at radiologist targets.
# Sample: sampled targets with suspicion score 3-5 and PSA density (B6_targets_final_qc.csv). Base model: suspicion score (categorical) +
# log PSA density; logistic regression, 5-fold GroupKFold by patient (cross-fitted predictions). Extended: + each model's agreement flag;
# + number of agreeing models (0-3). AUC; gain with a patient-cluster bootstrap 95% CI (2,000 resamples of patients). Overall and score 3.
#   python analysis/C5_target_level_discrimination.py --tag final_qc
import argparse, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='final_qc'); ap.add_argument('--C', type=float, default=1e6); a = ap.parse_args()
O = 'outputs/'
MODELS = [('3D_SegResNet', 'SegResNet'), ('nnUNet_official', 'nnU-Net'), ('ProViCNet', 'ProViCNet')]
T = pd.read_csv(O + f'B6_targets_{a.tag}.csv')
T = T[(T.path != 'unsampled') & T.score.isin([3, 4, 5]) & np.isfinite(T.lpsad)].reset_index(drop=True)
T['y'] = (T.path == 'csPCa').astype(int)
T['s4'], T['s5'] = (T.score == 4).astype(int), (T.score == 5).astype(int)
T['n_agree'] = T[[f'agree_{m}' for m, _ in MODELS]].sum(1)
print(f'targets {len(T)} (csPCa {T.y.sum()}) | score 3: {int((T.score == 3).sum())} (csPCa {int(T[T.score == 3].y.sum())}) | logistic C = {a.C}')


def cross_fit(D, cols):
    X, y, g = D[cols].to_numpy(float), D.y.to_numpy(int), D.Patient_ID.to_numpy()
    p = np.zeros(len(D))
    for tr, te in GroupKFold(5).split(X, y, g):
        p[te] = LogisticRegression(C=a.C, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def cluster_boot(D, p_new, p_base, B=2000, seed=11):
    rng = np.random.default_rng(seed)
    pats = D.Patient_ID.unique()
    idx = {q: np.where(D.Patient_ID.to_numpy() == q)[0] for q in pats}
    y = D.y.to_numpy(int); out = []
    for _ in range(B):
        i = np.concatenate([idx[q] for q in rng.choice(pats, len(pats))])
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_base[i]))
    return np.percentile(out, [2.5, 97.5])


rows = []
# models are fitted once on all targets (cross-fitted by patient); the score 3 results evaluate the same predictions within score 3 targets
BASE = ['s4', 's5', 'lpsad']
PRED = {'base': cross_fit(T, BASE)}
EXT = [(f'+ {n} agrees', f'agree_{m}') for m, n in MODELS] + [('+ number of agreeing models', 'n_agree')]
for lab, col in EXT:
    PRED[lab] = cross_fit(T, BASE + [col])
for scope, msk in (('all targets (score 3-5)', np.ones(len(T), bool)), ('score 3 targets', (T.score == 3).to_numpy())):
    D = T[msk].reset_index(drop=True)
    y = D.y.to_numpy(int)
    pb = PRED['base'][msk]
    ab = roc_auc_score(y, pb)
    print(f'\n[{scope}] n = {len(D)}, csPCa {y.sum()} | base (suspicion score + log PSA density) AUC {ab:.3f}')
    rows.append(dict(scope=scope, model='base: suspicion score + log PSA density', n=len(D), auc=ab, gain=0.0, lo=np.nan, hi=np.nan))
    for lab, col in EXT:
        pn = PRED[lab][msk]
        an = roc_auc_score(y, pn)
        lo, hi = cluster_boot(D, pn, pb)
        rows.append(dict(scope=scope, model=lab, n=len(D), auc=an, gain=an - ab, lo=lo, hi=hi))
        print(f'  {lab:30s} AUC {an:.3f} | gain {an - ab:+.3f} [patient bootstrap 95% CI {lo:+.3f}, {hi:+.3f}]')
    print('  csPCa by number of agreeing models: ' + ' | '.join(f'{k}: {int(D[D.n_agree == k].y.sum())}/{int((D.n_agree == k).sum())}' for k in range(4)))
pd.DataFrame(rows).to_csv(O + f'C5_results_{a.tag}.csv', index=False)
