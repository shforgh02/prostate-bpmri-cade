# Two descriptive analyses for the three models and the 757 examinations (no claim of added value is based on them).
#   1. Localization at a matched number of candidates: for each model all candidates of all examinations are ranked by peak probability
#      and the K = B x 757 highest are kept (B = 0.50, 0.75, 1.00, 1.50 candidates per examination). Outcome: evaluable cancers with a
#      kept candidate centre within 10 mm of a csPCa-positive core. 95% CIs and model differences by bootstrap of examinations (2,000).
#   2. Sampling of the top-3 candidates by biopsy cores: candidates within 10 mm of a radiologist target centre ("at a target") and all
#      other candidates ("away from targets"): number, share with at least one tracked core within 10 mm of the candidate centre,
#      median number of such cores, and share of sampled candidates with a csPCa-positive core within 10 mm.
#   python analysis/G1_burden_and_sampling.py   -> outputs/G1_burden_matched.csv, outputs/G1_candidate_sampling.csv
import os, sys, warnings
import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import label
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import SD, OUT, CC, p2s, target_units

ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
SESS = os.path.join(ROOT, 'MRI_US_Biopsy', 'Script', 'output', 'standardized_ucla_sessions_new')
ex = set(pd.read_csv(OUT + 'EXCLUDED_SESSIONS_v2.csv').Session_ID)
ep = pd.read_csv(SD + 'EXAMINATION_LEVEL_ENDPOINTS.csv')
ep = ep[~ep.Session_ID.isin(ex)].reset_index(drop=True)
cand = pd.read_csv(OUT + 'CANDIDATES_3MODELS.csv', low_memory=False)
cand = cand[(cand.Extractor == 'Official_RGA_Dynamic') & (cand.Is_Canonical == 1) & cand.Session_ID.isin(ep.Session_ID)].copy()
MODELS = [('3D_SegResNet', 'SegResNet'), ('nnUNet_official', 'nnU-Net'), ('ProViCNet', 'ProViCNet')]
cores = pd.read_csv(SD + 'CORES_REFERENCE.csv', low_memory=False)
cores = cores[cores.Session_ID_Matched.isin(ep.Session_ID)]
N = len(ep)
ev = ep[(ep.Is_csPCa == 1) & (ep.Has_Valid_3D_Trajectory == 1)].Session_ID.to_numpy()
print(f'examinations {N} | evaluable cancers {len(ev)}')

# ---------------------------------------------------------------- 1. matched number of candidates
sid = {s: i for i, s in enumerate(ep.Session_ID)}
evm = np.zeros(N, bool); evm[[sid[s] for s in ev]] = True
PER = {}
for m, nm in MODELS:
    q = cand[cand.Model == m]
    PER[m] = (q.Session_ID.map(sid).to_numpy(), q.Max_Prob.to_numpy(float), (q.CoM_Dist_csPCa_mm.to_numpy(float) <= 10))
    print(f'  {nm}: candidates {len(q)} ({len(q) / N:.2f} per examination)')


def sens(m, B, w):
    """w: how often each examination is drawn (bootstrap weights; ones = observed). Global budget K = B x N applied to the resample."""
    e, p, hit = PER[m]
    ww = w[e]
    o = np.argsort(-p, kind='stable')
    cum = np.cumsum(ww[o])
    keep = o[cum <= B * N] if (cum <= B * N).any() else o[:0]
    got = np.zeros(N, bool)
    got[e[keep][hit[keep] & (ww[keep] > 0)]] = True
    return (w * (got & evm)).sum() / (w * evm).sum(), ww[keep].sum() / N


rng = np.random.default_rng(20261115)
W = [np.bincount(rng.integers(0, N, N), minlength=N).astype(float) for _ in range(2000)]
one = np.ones(N)
rows = []
for B in (0.50, 0.75, 1.00, 1.50):
    bs = {}
    for m, nm in MODELS:
        s0, b0 = sens(m, B, one)
        bs[m] = np.array([sens(m, B, w)[0] for w in W])
        rows.append(dict(burden=B, model=nm, achieved_burden=b0, sensitivity=s0, lo=np.percentile(bs[m], 2.5), hi=np.percentile(bs[m], 97.5), n=len(ev), localized=int(round(s0 * len(ev)))))
        print(f'B = {B:.2f} | {nm:10s} achieved {b0:.2f} | localized {int(round(s0 * len(ev)))}/{len(ev)} = {s0:.1%} [{np.percentile(bs[m], 2.5):.1%}-{np.percentile(bs[m], 97.5):.1%}]')
    for (m1, n1), (m2, n2) in ((MODELS[0], MODELS[1]), (MODELS[0], MODELS[2]), (MODELS[1], MODELS[2])):
        d = bs[m1] - bs[m2]
        d0 = sens(m1, B, one)[0] - sens(m2, B, one)[0]
        rows.append(dict(burden=B, model=f'{n1} minus {n2}', sensitivity=d0, lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
        print(f'           {n1} minus {n2}: {d0:+.1%} [{np.percentile(d, 2.5):+.1%}, {np.percentile(d, 97.5):+.1%}]')
pd.DataFrame(rows).to_csv(OUT + 'G1_burden_matched.csv', index=False)

# ---------------------------------------------------------------- 2. sampling of top-3 candidates by cores
cv = cores[cores.Has_Valid_Physical_Trajectory == 1]
GRP = {s: g for s, g in cv.groupby('Session_ID_Matched')}
TC = {}
for s in ep.Session_ID:
    f = os.path.join(SESS, s, 'target_1_mask.nii.gz')
    cen = []
    if os.path.exists(f):
        nii = nib.load(f)
        lbl, _ = label(np.asarray(nii.dataobj) > 0)
        for u in target_units(s, lbl):
            com = np.argwhere(u['mask']).mean(0)
            cen.append((nii.affine[:3, :3] @ com + nii.affine[:3, 3]) * np.array([-1.0, -1.0, 1.0]))
    TC[s] = np.array(cen) if cen else np.zeros((0, 3))
top3 = cand[cand.Candidate_Rank <= 3]
rows = []
for m, nm in MODELS:
    q = top3[top3.Model == m]
    rec = []
    for s, g in q.groupby('Session_ID'):
        P = g[['CoM_LPS_X_mm', 'CoM_LPS_Y_mm', 'CoM_LPS_Z_mm']].to_numpy(float)
        at = (np.sqrt(((P[:, None, :] - TC[s][None]) ** 2).sum(2)).min(1) <= 10) if len(TC[s]) else np.zeros(len(P), bool)
        c_ = GRP.get(s)
        if c_ is not None:
            d = p2s(P, c_[CC[:3]].to_numpy(float), c_[CC[3:]].to_numpy(float)) <= 10
            ncore = d.sum(1); cs = (d & (c_.Is_csPCa.to_numpy(int) == 1)[None]).any(1)
        else:
            ncore = np.zeros(len(P), int); cs = np.zeros(len(P), bool)
        rec += [dict(at_target=bool(a), ncore=int(n_), cs=bool(x)) for a, n_, x in zip(at, ncore, cs)]
    R = pd.DataFrame(rec)
    for lab, sub in (('all top-3 candidates', R), ('at a radiologist target', R[R.at_target]), ('away from radiologist targets', R[~R.at_target])):
        smp = sub[sub.ncore > 0]
        rows.append(dict(model=nm, group=lab, n=len(sub), share=len(sub) / len(R), sampled=len(smp), median_cores=float(sub.ncore.median()), cspca=int(smp.cs.sum())))
        print(f'{nm:10s} {lab:30s} n={len(sub):5d} ({len(sub) / len(R):5.1%}) | sampled by a core {len(smp)} ({len(smp) / max(len(sub), 1):5.1%}), median cores {sub.ncore.median():.0f} | '
              f'csPCa among sampled {int(smp.cs.sum())}/{len(smp)} = {smp.cs.mean():5.1%}')
pd.DataFrame(rows).to_csv(OUT + 'G1_candidate_sampling.csv', index=False)
