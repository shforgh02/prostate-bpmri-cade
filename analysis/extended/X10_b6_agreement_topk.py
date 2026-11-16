# X10: copy of analysis/B6_per_target_agreement.py with the number of candidates per examination taken from --topk
# (1, 2, 3 or 5; the main analysis uses 3). Agreement = one of the k highest-ranked candidate centres within 10 mm of the target centre.
# Same targets, histology, model and covariates as B6. Outputs go to outputs/extended/.
#   python analysis/extended/X10_b6_agreement_topk.py --exclude outputs/EXCLUDED_SESSIONS_v2.csv --candidates outputs/CANDIDATES_3MODELS.csv --tag final_qc --topk 1
import sys, os, warnings
import numpy as np, pandas as pd, nibabel as nib
import statsmodels.api as sm
from scipy.ndimage import label
warnings.filterwarnings("ignore")
sys.path.insert(0, 'analysis')
from _common import SD, OUT, CC, args, load, p2s, cand_xyz, target_units, gee_fit

a = args()
ep, cand, cores, ms, models = load(a)
ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
SESS = os.path.join(ROOT, 'MRI_US_Biopsy', 'Script', 'output', 'standardized_ucla_sessions_new')
cv = cores[cores.Has_Valid_Physical_Trajectory == 1]
grp = {s: g for s, g in cv.groupby('Session_ID_Matched')}
CX = {m: cand_xyz(cand, m) for m in models}
XO = 'outputs/extended/'
os.makedirs(XO, exist_ok=True)
cache = XO + f'X10_targets_top{a.topk}.csv'
rows = []
for s, pid, score in zip(ep.Session_ID, ep.Patient_ID, ep.Exact_Exam_UCLA_Score):
    f = os.path.join(SESS, s, 'target_1_mask.nii.gz')
    if not os.path.exists(f):
        continue
    nii = nib.load(f)
    lbl, n = label(np.asarray(nii.dataobj) > 0)
    units = target_units(s, lbl)                                                  # one unit per radiologist target surface, with its own score
    n = len(units)
    g = grp.get(s)
    for u in units:
        rc = u['tno']
        com = np.argwhere(u['mask']).mean(0)
        lps = (nii.affine[:3, :3] @ com + nii.affine[:3, 3]) * np.array([-1.0, -1.0, 1.0])
        d = {'cs': np.inf, 'gg1': np.inf, 'ben': np.inf}
        if g is not None:
            dist = p2s(lps[None], g[CC[:3]].to_numpy(float), g[CC[3:]].to_numpy(float))[0]
            isup = g.ISUP_Grade.fillna(0).to_numpy(float)
            cs = g.Is_csPCa.to_numpy(int) == 1
            for key, m in (('cs', cs), ('gg1', (~cs) & (isup == 1)), ('ben', (~cs) & (isup != 1))):
                if m.any():
                    d[key] = dist[m].min()
        path = 'csPCa' if d['cs'] <= 10 else 'GG1' if d['gg1'] <= 10 else 'benign' if d['ben'] <= 10 else 'unsampled'
        r = dict(Session_ID=s, Patient_ID=pid, target=rc, n_targets=n, vol_cc=float(u['mask'].sum()) * 0.001, exam_score=score, score=u['score'], score_conflict=u['conflict'], path=path)
        for m in models:
            c = CX[m].get(s)
            r['agree_' + m] = int(c is not None and np.sqrt(((c - lps[None]) ** 2).sum(1)).min() <= 10)
        rows.append(r)
T = pd.DataFrame(rows)
T = T.merge(ms[['PSA', 'MRI_Prostate_Vol_cc']].reset_index(), on='Session_ID', how='left')
T['lpsad'] = np.log(T.PSA / T.MRI_Prostate_Vol_cc)
T.to_csv(cache, index=False)
print(f'cohort {len(ep)} exams | targets {len(T)} | sampled {(T.path != "unsampled").sum()} | csPCa {(T.path == "csPCa").sum()} | GG1 {(T.path == "GG1").sum()} | benign {(T.path == "benign").sum()}')
S = T[(T.path != 'unsampled') & T.score.isin([3, 4, 5]) & np.isfinite(T.lpsad)].copy()
S['y'] = (S.path == 'csPCa').astype(int)
out = []


def fit(sub, col, with_score):
    X = pd.DataFrame({'agree': sub[col].to_numpy(float), 'lpsad': sub.lpsad.to_numpy(float)})
    sc = sub.exam_score if with_score == 'exam' else sub.score                   # 'exam': adjustment used before the target scores were linked
    if with_score:
        X['s4'] = (sc == 4).to_numpy(float)
        X['s5'] = (sc == 5).to_numpy(float)
    r = gee_fit(sub.y.to_numpy(), sm.add_constant(X), sub.Patient_ID.to_numpy())
    ci = r.conf_int().loc['agree']
    return np.exp(r.params['agree']), np.exp(ci[0]), np.exp(ci[1]), r.pvalues['agree']


SUBSETS = (('all sampled targets, score 3-5', S, True), ('score 3', S[S.score == 3], False), ('score 4', S[S.score == 4], False), ('score 5', S[S.score == 5], False),
           ('single-target exams, score 3-5', S[S.n_targets == 1], True), ('single-target exams, score 3', S[(S.n_targets == 1) & (S.score == 3)], False),
           ('sensitivity: adjusted for the examination-level score', S, 'exam'),
           ('sensitivity: targets without conflicting scores, score 3-5', S[S.score_conflict == 0], True),
           ('sensitivity: targets without conflicting scores, score 3', S[(S.score_conflict == 0) & (S.score == 3)], False))
for m in models:
    col = 'agree_' + m
    print(f'\n[{m}] agreement in {S[col].mean():.1%} of {len(S)} sampled targets')
    for lab, sub, ws in SUBSETS:
        y1, y0 = sub.y[sub[col] == 1], sub.y[sub[col] == 0]
        try:
            o, lo, hi, p = fit(sub, col, ws)
        except Exception as e:
            o = lo = hi = p = np.nan
        print(f'  {lab:32s} n={len(sub):4d} | csPCa with agreement {y1.mean():5.1%} ({y1.sum()}/{len(y1)}) vs without {y0.mean():5.1%} ({y0.sum()}/{len(y0)}) | adjusted OR {o:.2f} [{lo:.2f}-{hi:.2f}] p={p:.4f}')
        out.append(dict(model=m, subset=lab, n=len(sub), agree_n=len(y1), agree_cspca=int(y1.sum()), disagree_n=len(y0), disagree_cspca=int(y0.sum()), OR=o, lo=lo, hi=hi, p=p))
pd.DataFrame(out).to_csv(XO + f'X10_results_top{a.topk}.csv', index=False)
