# Examination level: each model's score (highest candidate confidence; 0 if no candidate) alone and added to
# suspicion score + PSA density. Any number of models. (The agreement analysis is now per target: B6_per_target_agreement.py.)
# Run from the repository root:
#   python analysis/A3_agreement_and_psa_density.py [--exclude csv] [--candidates csv] [--tag name]
import sys, os, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SD, OUT, args, load

a = args()
ep, cand, cores, ms, models = load(a)
D = ep.merge(ms[['PSA', 'MRI_Prostate_Vol_cc']].reset_index(), on='Session_ID', how='left')
D['lpsad'] = np.log(D.PSA / D.MRI_Prostate_Vol_cc)
for m in models:
    D = D.merge(cand[cand.Model == m].groupby('Session_ID').Max_Prob.max().rename('s_' + m).reset_index(), on='Session_ID', how='left')
    D['s_' + m] = D['s_' + m].fillna(0.0)
rng = np.random.default_rng(1)


def cross_fit(y, X):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        p[te] = LogisticRegression(C=1e6, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def boot_ci(y, p1, p0=None, B=2000):
    o = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        o.append(roc_auc_score(y[i], p1[i]) - (roc_auc_score(y[i], p0[i]) if p0 is not None else 0))
    return np.percentile(o, [2.5, 97.5])


out = []
y_all = D.Is_csPCa.to_numpy(int)
print(f'cohort {len(D)} exams, csPCa {y_all.sum()} | models {models}')
print('\n-- each score alone, all exams --')
for m in models:
    s = D['s_' + m].to_numpy(float)
    lo, hi = boot_ci(y_all, s)
    print(f'  {m:22s} AUROC {roc_auc_score(y_all, s):.3f} [{lo:.3f}-{hi:.3f}]')
    out.append(dict(model=m, analysis='AUROC alone, all exams', n=len(D), value=roc_auc_score(y_all, s), lo=lo, hi=hi))
E = D[np.isfinite(D.lpsad) & D.Exact_Exam_UCLA_Score.isin([3, 4, 5])].copy()
y = E.Is_csPCa.to_numpy(int)
U = pd.get_dummies(E.Exact_Exam_UCLA_Score.astype(int), drop_first=True).to_numpy(float)
base = np.column_stack([U, E.lpsad])
pb = cross_fit(y, base)
print(f'\n-- suspicion score 3-5 with PSA density: n={len(E)}, csPCa {y.sum()} --')
print(f'  suspicion score {roc_auc_score(y, E.Exact_Exam_UCLA_Score):.3f} | PSA density {roc_auc_score(y, E.lpsad):.3f} | score + PSA density (cross-fitted) {roc_auc_score(y, pb):.3f}')
out.append(dict(model='reference', analysis='suspicion score', n=len(E), value=roc_auc_score(y, E.Exact_Exam_UCLA_Score)))
out.append(dict(model='reference', analysis='PSA density', n=len(E), value=roc_auc_score(y, E.lpsad)))
out.append(dict(model='reference', analysis='score + PSA density', n=len(E), value=roc_auc_score(y, pb)))
ll0 = sm.Logit(y, sm.add_constant(base)).fit(disp=0).llf
for m in models:
    s = E['s_' + m].to_numpy(float)
    X = np.column_stack([base, s])
    pf = cross_fit(y, X)
    lr = chi2.sf(2 * (sm.Logit(y, sm.add_constant(X)).fit(disp=0).llf - ll0), 1)
    lo, hi = boot_ci(y, pf, pb)
    g = roc_auc_score(y, pf) - roc_auc_score(y, pb)
    print(f'  + {m:22s} alone {roc_auc_score(y, s):.3f} | added: {roc_auc_score(y, pf):.3f} (gain {g:+.3f} [{lo:+.3f}, {hi:+.3f}]; likelihood-ratio p={lr:.3f})')
    out.append(dict(model=m, analysis='gain over score + PSA density', n=len(E), value=g, lo=lo, hi=hi, p=lr))
pd.DataFrame(out).to_csv(OUT + f'A3_results_{a.tag}.csv', index=False)
