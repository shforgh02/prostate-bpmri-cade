# Needle-supervised readout of frozen ProViCNet features.
# Question: if a small model is trained on the histology of the needles themselves (not on radiologist outlines), does it
# recognise cancer in OFF-TARGET cores better than ProViCNet's own output?
#
#   python analysis/C2_needle_probe.py --make-cores                 writes outputs/cores_for_modal.csv (step before B8)
#   python analysis/C2_needle_probe.py --feats <folder of <case>.npz> --exclude outputs/EXCLUDED_SESSIONS_v2.csv
#
# Settings (no tuning): 5-fold cross-validation by patient; standardise -> PCA 64 -> logistic regression (C = 1.0, balanced);
# training on ALL cores (targeted + systematic) of the training patients; evaluation on off-target systematic cores of test patients.
import os, sys, glob, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SD, OUT, CC, p2s, cand_xyz, bbox

ap = argparse.ArgumentParser()
ap.add_argument('--make-cores', action='store_true')
ap.add_argument('--feats')
ap.add_argument('--exclude', default=OUT + 'EXCLUDED_SESSIONS_v2.csv')
ap.add_argument('--candidates', default=OUT + 'CANDIDATES_3MODELS.csv')
ap.add_argument('--tag', default='stage2')
a = ap.parse_args()
ep = pd.read_csv(SD + 'EXAMINATION_LEVEL_ENDPOINTS.csv')
cores = pd.read_csv(SD + 'CORES_REFERENCE.csv', low_memory=False)
cores = cores[(cores.Has_Valid_Physical_Trajectory == 1) & cores.Session_ID_Matched.isin(ep.Session_ID)]
if a.make_cores:
    c = cores[['Session_ID_Matched', 'Original_Row_ID'] + CC].copy()
    c.columns = ['Session_ID', 'Original_Row_ID', 'bx', 'by', 'bz', 'tx', 'ty', 'tz']
    c.to_csv(OUT + 'cores_for_modal.csv', index=False)
    print('wrote', OUT + 'cores_for_modal.csv', '| cores', len(c), '| sessions', c.Session_ID.nunique())
    sys.exit(0)

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

excl = set(pd.read_csv(a.exclude).Session_ID)
ep = ep[~ep.Session_ID.isin(excl)]
ms = pd.read_csv('sources/Phase3_Patient_Session_Level_Master.csv', low_memory=False).set_index('Session_ID')
# ---- features ---------------------------------------------------------------------------------------------------------
ids, F, P, NV = [], [], [], []
for f in sorted(glob.glob(os.path.join(a.feats, '*.npz'))):
    sid = os.path.basename(f)[5:-4]
    if sid not in set(ep.Session_ID):
        continue
    d = np.load(f)
    ids.append(d['core_id']); F.append(d['feats'].astype(np.float32)); P.append(d['probs']); NV.append(d['nvalid'])
ids, F, P, NV = np.concatenate(ids), np.concatenate(F), np.concatenate(P), np.concatenate(NV)
T = cores.set_index('Original_Row_ID').loc[ids].reset_index()
T['own_p'] = P[:, 0]; T['own_p_any'] = P[:, 2]; T['nvalid'] = NV
keep = (T.nvalid >= 5).to_numpy()
T, F = T[keep].reset_index(drop=True), F[keep]
T = T.merge(ep[['Session_ID', 'Patient_ID', 'Is_csPCa']].rename(columns={'Session_ID': 'Session_ID_Matched', 'Is_csPCa': 'exam_cs'}), on='Session_ID_Matched')
print(f'exams with features {T.Session_ID_Matched.nunique()} | cores {len(T)} | csPCa cores {int(T.Is_csPCa.sum())}')
# ---- off-target flag (same definition as A2) and distance to nearest ProViCNet candidate -----------------------------------
cand = pd.read_csv(a.candidates, low_memory=False)
cand = cand[(cand.Extractor == 'Official_RGA_Dynamic') & (cand.Candidate_Rank <= 3)]
CX = cand_xyz(cand, 'ProViCNet')
T['near_rad'] = 0; T['d_cand'] = 99.0; T[['px', 'py', 'pz']] = np.nan
for s, g in T.groupby('Session_ID_Matched'):
    A, B = g[CC[:3]].to_numpy(float), g[CC[3:]].to_numpy(float)
    near = np.zeros(len(g), bool)
    tg = g[g.Is_Targeted_Core == 1]
    if len(tg):
        TA, TB = tg[CC[:3]].to_numpy(float), tg[CC[3:]].to_numpy(float)
        near |= p2s(np.concatenate([TA + (TB - TA) * f for f in np.linspace(0, 1, 9)]), A, B).min(0) <= 10
    h = ms.loc[s, ['Human_Rad_CoM_LPS_X_mm', 'Human_Rad_CoM_LPS_Y_mm', 'Human_Rad_CoM_LPS_Z_mm']].to_numpy(float)
    if np.isfinite(h).all():
        near |= p2s(h[None], A, B)[0] <= 10
    T.loc[g.index, 'near_rad'] = near.astype(int)
    if s in CX:
        T.loc[g.index, 'd_cand'] = p2s(CX[s], A, B).min(0)
    mn, sp = bbox(ms, s)
    T.loc[g.index, ['px', 'py', 'pz']] = ((A + B) / 2 - mn) / sp
T['plat'] = (T.px - 0.5).abs()
y = T.Is_csPCa.to_numpy(int)
grp = T.Patient_ID.to_numpy()
# ---- cross-validated scores ---------------------------------------------------------------------------------------------------
def oof(X, pca=True):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        steps = [StandardScaler()] + ([PCA(64, random_state=0)] if pca else []) + [LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)]
        p[te] = make_pipeline(*steps).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p
pos = T[['px', 'py', 'pz', 'plat']].fillna(0.5).to_numpy(float)
SC = {'ProViCNet own csPCa probability along the needle': T.own_p.fillna(0).to_numpy(),
      'distance to nearest ProViCNet candidate (as in A2)': -T.d_cand.to_numpy(),
      'position only (anatomical prior)': oof(pos, pca=False),
      'NEEDLE-SUPERVISED READOUT (frozen features)': oof(F),
      'needle-supervised readout + position': oof(np.column_stack([F, pos]))}
off = ((T.Is_Systematic_Core == 1) & (T.near_rad == 0)).to_numpy()
SETS = {'all cores': np.ones(len(T), bool), 'systematic cores': (T.Is_Systematic_Core == 1).to_numpy(), 'OFF-TARGET systematic': off,
        'OFF-TARGET, csPCa patients': off & (T.exam_cs == 1).to_numpy()}
for k, m in SETS.items():
    print(f'{k}: n={m.sum()}, csPCa+ {y[m].sum()} ({y[m].mean():.1%})')
print('\npooled AUROC'); print(f'{"score":54s}' + ''.join(f'{k[:26]:>28s}' for k in SETS))
for nm, s in SC.items():
    print(f'{nm:54s}' + ''.join(f'{roc_auc_score(y[m], s[m]):28.3f}' for m in SETS.values()))
# ---- within-patient AUROC on off-target cores (main comparison of this readout) --------------------------------------------------------------
sess = T.Session_ID_Matched.to_numpy()
ELIG = [np.where((sess == s) & off)[0] for s in np.unique(sess)]
ELIG = [k for k in ELIG if 0 < y[k].sum() < len(k)]
W = {nm: np.array([roc_auc_score(y[k], s[k]) for k in ELIG]) for nm, s in SC.items()}
rng = np.random.default_rng(11)
B = rng.integers(0, len(ELIG), (5000, len(ELIG)))
rows = []
print(f'\nwithin-patient AUROC, off-target cores ({len(ELIG)} patients with at least one positive and one negative off-target core)')
ref = 'ProViCNet own csPCa probability along the needle'
for nm, w in W.items():
    lo, hi = np.percentile(w[B].mean(1), [2.5, 97.5])
    d = (w - W[ref])[B].mean(1)
    dl, dh = np.percentile(d, [2.5, 97.5])
    pd_ = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f'  {nm:54s} {w.mean():.3f} [{lo:.3f}-{hi:.3f}] | difference vs own probability {w.mean() - W[ref].mean():+.3f} [{dl:+.3f}, {dh:+.3f}] p={pd_:.3f}')
    rows.append(dict(score=nm, within_auroc=w.mean(), lo=lo, hi=hi, diff_vs_own=w.mean() - W[ref].mean(), diff_lo=dl, diff_hi=dh, p=pd_,
                     pooled_off_target=roc_auc_score(y[off], SC[nm][off]), pooled_off_target_cs=roc_auc_score(y[SETS['OFF-TARGET, csPCa patients']], SC[nm][SETS['OFF-TARGET, csPCa patients']])))
pd.DataFrame(rows).to_csv(OUT + f'C2_results_{a.tag}.csv', index=False)
