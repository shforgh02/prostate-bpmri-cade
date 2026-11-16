# Candidate extraction for ANY model whose probability maps are stored as <Session_ID>.npz with keys
#   heatmap (x,y,z float, session 1 mm grid, already zero outside the dilated prostate), affine (RAS 4x4), pros_mask.
# Same logic as pipeline/04_extract_lesion_candidates_unified.py, "Official_RGA_Dynamic" block (lines 398-474).
# Run from the repository root:
#   python analysis/B5_extract_candidates_generic.py --model NAME --dir <npz folder> --out <csv> [--exclude csv] [--regress]
# --regress compares the result with the existing rows of that model name in CANDIDATES_ANALYSIS_READY.csv.
import os, sys, argparse, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from report_guided_annotation import extract_lesion_candidates
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--dir', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--exclude'); ap.add_argument('--regress', action='store_true'); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
sd = 'results/tables/'
ep = pd.read_csv(sd+'EXAMINATION_LEVEL_ENDPOINTS.csv'); cores = pd.read_csv(sd+'CORES_REFERENCE.csv', low_memory=False)
if a.exclude: ep = ep[~ep.Session_ID.isin(pd.read_csv(a.exclude).Session_ID)]
cc = ['Bx Base X (MRI Coord)','Bx Base Y (MRI Coord)','Bx Base Z (MRI Coord)','Bx Tip X (MRI Coord)','Bx Tip Y (MRI Coord)','Bx Tip Z (MRI Coord)']
pos = cores[cores.Is_Valid_csPCa_Trajectory == 1]
SEG = {s: (g[cc[:3]].to_numpy(float), g[cc[3:]].to_numpy(float)) for s, g in pos.groupby('Session_ID_Matched')}
def vox_to_lps(v, aff):
    ras = aff[:3, :3] @ np.asarray(v, float) + aff[:3, 3]; return ras * np.array([-1.0, -1.0, 1.0])
def p2seg(p, A, B):
    AB = B - A; L2 = (AB**2).sum(1); L2[L2 == 0] = 1e-9
    t = np.clip(((p[None] - A) * AB).sum(1) / L2, 0, 1); return float(np.sqrt(((p[None] - (A + t[:, None] * AB))**2).sum(1)).min())
rows = []; missing = 0; todo = list(zip(ep.Session_ID, ep.Patient_ID))
if a.limit: todo = todo[:a.limit]
for n, (sid, pid) in enumerate(todo):
    f = os.path.join(a.dir, sid + '.npz')
    if not os.path.exists(f): missing += 1; continue
    d = np.load(f); prob = d['heatmap'].astype(np.float32); aff = d['affine'].astype(float)
    hard, conf, idx = extract_lesion_candidates(prob.astype(np.float64), threshold='dynamic-fast', num_lesions_to_extract=5, min_voxels_detection=30, dynamic_threshold_factor=2.5)
    cands = []
    for l_id, c in conf:
        m = (idx == l_id); nv = int(m.sum())
        if nv < 30: continue
        cands.append(dict(l_id=l_id, conf=float(c), vol=nv * 0.001, n=nv, com=np.argwhere(m).mean(0)))
    cands.sort(key=lambda x: (-x['conf'], -x['vol'], x['l_id']))
    for rank, c in enumerate(cands[:5], start=1):
        lps = vox_to_lps(c['com'], aff); dcs = p2seg(lps, *SEG[sid]) if sid in SEG else np.nan
        rows.append(dict(Candidate_UID=f'{a.model}_RGA_{sid}_cand{rank}', Model=a.model, Extractor='Official_RGA_Dynamic', Session_ID=sid, Patient_ID=pid, Is_Canonical=1, Candidate_Rank=rank,
                         Max_Prob=round(c['conf'], 4), Volume_cc=round(c['vol'], 3), Voxel_Count=c['n'], CoM_LPS_X_mm=round(lps[0], 3), CoM_LPS_Y_mm=round(lps[1], 3), CoM_LPS_Z_mm=round(lps[2], 3),
                         CoM_Dist_csPCa_mm=round(dcs, 3) if np.isfinite(dcs) else np.nan))
    if n % 200 == 0: print(n, sid, flush=True)
R = pd.DataFrame(rows); R.to_csv(a.out, index=False)
print(f'{a.model}: exams {len(todo)} | maps missing {missing} | candidates {len(R)} | top-3 {int((R.Candidate_Rank<=3).sum())}')
if a.regress:
    old = pd.read_csv(sd+'CANDIDATES_ANALYSIS_READY.csv'); old = old[(old.Model == a.model) & (old.Extractor == 'Official_RGA_Dynamic') & (old.Is_Canonical == 1) & old.Session_ID.isin([s for s, _ in todo])]
    m = old.merge(R, on=['Session_ID', 'Candidate_Rank'], how='outer', suffixes=('_old', '_new'), indicator=True)
    both = m[m._merge == 'both']; dd = np.sqrt(sum((both[f'CoM_LPS_{k}_mm_old'] - both[f'CoM_LPS_{k}_mm_new'])**2 for k in 'XYZ'))
    print(f'REGRESSION: old rows {len(old)} | new rows {len(R)} | matched {len(both)} | only old {(m._merge=="left_only").sum()} | only new {(m._merge=="right_only").sum()} | max centre difference {dd.max():.4f} mm | max |prob diff| {(both.Max_Prob_old-both.Max_Prob_new).abs().max():.4f}')
    ok = len(old) == len(R) == len(both) and dd.max() <= 0.01
    print('REGRESSION', 'PASSED' if ok else 'FAILED'); sys.exit(0 if ok else 1)
