# PROSTATEx replication, step 4: ProViCNet probability maps of the PROSTATEx examinations -> candidates, exactly as for the UCLA cohort
#   (scripts B3 + B5): map resampled (linear) onto the 1-mm grid, set to zero outside the gland mask dilated by 4 voxels, candidates with
#   report-guided annotation (dynamic-fast, up to 5 lesions, at least 30 voxels, dynamic threshold factor 2.5), ranked by peak probability,
#   then volume; centre of mass in patient (LPS) coordinates. Outcome-blind (no finding label is read).
#   python analysis/R4_prostatex_candidates.py
import os, warnings
import numpy as np, pandas as pd, SimpleITK as sitk
from scipy.ndimage import binary_dilation
warnings.filterwarnings("ignore")
from report_guided_annotation import extract_lesion_candidates

REP = 'outputs/replication/'
ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
D = os.path.join(ROOT, 'PROSTATEx_replication')
M = pd.read_csv(REP + 'match_table_R1b.csv')
rows, miss = [], []
for n, c in enumerate(M.ProxID):
    f = os.path.join(D, 'pred', f'PX_{c}.nii.gz')
    if not os.path.exists(f):
        miss.append(c)
        continue
    pm = sitk.ReadImage(os.path.join(D, 'grid1mm', f'{c}_prostate_mask.nii.gz'))
    prob = sitk.Cast(sitk.ReadImage(f), sitk.sitkFloat32)
    res = sitk.Resample(prob, pm, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    heat = np.clip(sitk.GetArrayFromImage(res).transpose(2, 1, 0), 0, 1)           # (x, y, z)
    mask = sitk.GetArrayFromImage(pm).transpose(2, 1, 0) > 0
    heat[~binary_dilation(mask, iterations=4)] = 0.0
    hard, conf, idx = extract_lesion_candidates(heat.astype(np.float64), threshold='dynamic-fast', num_lesions_to_extract=5, min_voxels_detection=30, dynamic_threshold_factor=2.5)
    cands = []
    for l_id, cf in conf:
        m = idx == l_id
        nv = int(m.sum())
        if nv < 30:
            continue
        cands.append(dict(l_id=l_id, conf=float(cf), n=nv, com=np.argwhere(m).mean(0)))
    cands.sort(key=lambda v: (-v['conf'], -v['n'], v['l_id']))
    for rank, v in enumerate(cands[:5], start=1):
        lps = pm.TransformContinuousIndexToPhysicalPoint([float(t) for t in v['com']])
        rows.append(dict(ProxID=c, Candidate_Rank=rank, Max_Prob=round(v['conf'], 4), Volume_cc=round(v['n'] * 0.001, 3),
                         CoM_LPS_X_mm=round(lps[0], 3), CoM_LPS_Y_mm=round(lps[1], 3), CoM_LPS_Z_mm=round(lps[2], 3)))
    if not cands:
        rows.append(dict(ProxID=c, Candidate_Rank=0, Max_Prob=float(heat.max())))    # examination without a candidate (kept; counts as no agreement)
    if n % 40 == 0:
        print(n, c, flush=True)
R = pd.DataFrame(rows)
R.to_csv(REP + 'R4_candidates_ProViCNet.csv', index=False)
k = R[R.Candidate_Rank > 0]
per = k.groupby('ProxID').size().reindex(M.ProxID[~M.ProxID.isin(miss)]).fillna(0)
print(f'examinations {len(M)} | probability maps missing {len(miss)} | with at least one candidate {int((per > 0).sum())} | none {int((per == 0).sum())} | '
      f'candidates {len(k)} | top-3 {int((k.Candidate_Rank <= 3).sum())} | mean number per examination {per.mean():.2f}')
