# Links every radiologist target (connected component of target_1_mask.nii.gz, the unit of B6) to its own suspicion score.
# The public file Target-Data_2019-12-05-2.xlsx holds one row per target and biopsy session: Patient ID, MRI series, "Target No.", ROI volume, UCLA score.
# The public surface files are named <patient>-Target<No>-seriesUID-<MRI series>.STL, so the target number of every surface is known.
# Link: the vertices of each target surface are mapped into the session grid and assigned to the mask component they fall in (nearest component within 3 mm).
# Checks written to the output: surface volume against the ROI volume of the spreadsheet; share of vertices in the assigned component.
# Rule for conflicting scores (same MRI target listed with different scores in different biopsy sessions): the highest score, which is the rule
#   that pipeline/03 already used for the examination-level score (maximum over all rows of the MRI series). Conflicts are flagged for a sensitivity analysis.
# Outcome-blind: uses surfaces, masks and the target spreadsheet only.
#   python analysis/B6c_link_target_scores.py   -> outputs/TARGET_SCORES.csv
import os, glob, re, warnings
import numpy as np, pandas as pd, nibabel as nib, trimesh
from scipy.ndimage import label, distance_transform_edt
warnings.filterwarnings("ignore")

ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
UCLA = os.path.join(ROOT, 'MRI_US_Biopsy')
SESS = os.path.join(UCLA, 'Script', 'output', 'standardized_ucla_sessions_new')
STL = os.path.join(UCLA, 'PKG - Prostate-MRI-US-Biopsy-STL', 'Prostate-MRI-US-Biopsy', 'STLs', 'STLs')
ep = pd.read_csv('results/tables/EXAMINATION_LEVEL_ENDPOINTS.csv')
x = pd.read_excel(os.path.join(UCLA, 'Target-Data_2019-12-05-2.xlsx'))
x.columns = ['score', 'roi_vol', 'tno', 'us', 'mr', 'pid']
x['mr'] = x.mr.astype(str).str.strip()
SC = x.groupby(['pid', 'mr', 'tno']).agg(smax=('score', 'max'), smin=('score', 'min'), n_rows=('score', 'size'), roi_vol=('roi_vol', 'first'),
                                         roi_vol_sd=('roi_vol', lambda v: float(np.std(v)))).reset_index()
rows, notes = [], []
for s, pid, escore in zip(ep.Session_ID, ep.Patient_ID, ep.Exact_Exam_UCLA_Score):
    f = os.path.join(SESS, s, 'target_1_mask.nii.gz')
    if not os.path.exists(f):
        continue
    nii = nib.load(f)
    lbl, n = label(np.asarray(nii.dataobj) > 0)
    if n == 0:
        continue
    mrs = [m for m in x[x.pid == pid].mr.unique() if m.endswith(s.split('_')[-1])]
    if len(mrs) != 1:
        notes.append((s, f'MRI series not unique in the target spreadsheet ({len(mrs)})'))
        for rc in range(1, n + 1):
            rows.append(dict(Session_ID=s, target=rc, exam_score=escore))
        continue
    mr = mrs[0]
    dist, idx = distance_transform_edt(lbl == 0, return_indices=True)            # nearest labelled voxel for every voxel
    inv = np.linalg.inv(nii.affine)
    link = {rc: [] for rc in range(1, n + 1)}
    for stl in sorted(glob.glob(os.path.join(STL, f'{pid}-Target*-seriesUID-{mr}.STL'))):
        tno = int(re.search(r'-Target(\d+)-', os.path.basename(stl)).group(1))
        mesh = trimesh.load(stl)
        v = np.asarray(mesh.vertices, float)
        pts = np.vstack([v, v.mean(0, keepdims=True)]) * np.array([-1.0, -1.0, 1.0])          # LPS -> RAS
        ijk = np.rint(pts @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(lbl.shape)), axis=1)
        ijk = ijk[ok]
        near = dist[ijk[:, 0], ijk[:, 1], ijk[:, 2]] <= 3.0
        lab = lbl[idx[0][ijk[:, 0], ijk[:, 1], ijk[:, 2]], idx[1][ijk[:, 0], ijk[:, 1], ijk[:, 2]], idx[2][ijk[:, 0], ijk[:, 1], ijk[:, 2]]][near]
        if len(lab) == 0:
            notes.append((s, f'Target{tno}: surface does not touch the mask'))
            continue
        cnt = np.bincount(lab, minlength=n + 1)
        sc = SC[(SC.pid == pid) & (SC.mr == mr) & (SC.tno == tno)]
        if len(sc) != 1:
            notes.append((s, f'Target{tno}: {len(sc)} rows in the target spreadsheet'))
            continue
        sc = sc.iloc[0]
        for rc in range(1, n + 1):
            if cnt[rc] / len(v) >= 0.20:                                          # a surface may span two components (split) or share one (merged)
                link[rc].append(dict(tno=tno, frac=cnt[rc] / max(len(lab), 1), smax=int(sc.smax), smin=int(sc.smin), stl_vol=abs(float(mesh.volume)) / 1000.0,
                                     roi_vol=float(sc.roi_vol), n_rows=int(sc.n_rows)))
    for rc in range(1, n + 1):
        L = link[rc]
        r = dict(Session_ID=s, target=rc, exam_score=escore, n_surfaces=len(L))
        if L:
            r.update(target_nos='+'.join(str(l['tno']) for l in L), target_score=max(l['smax'] for l in L), target_score_min=min(l['smin'] for l in L),
                     vertex_frac=min(l['frac'] for l in L), stl_vol_cc=sum(l['stl_vol'] for l in L), roi_vol_cc=sum(l['roi_vol'] for l in L),
                     mask_vol_cc=float((lbl == rc).sum()) * 0.001)
            r['score_conflict'] = int(r['target_score'] != r['target_score_min'])
        rows.append(r)
T = pd.DataFrame(rows)
T.to_csv('outputs/TARGET_SCORES.csv', index=False)
pd.DataFrame(notes, columns=['Session_ID', 'note']).to_csv('outputs/TARGET_SCORES_notes.csv', index=False)
k = T.target_score.notna()
print(f'components {len(T)} in {T.Session_ID.nunique()} examinations | linked {int(k.sum())} | not linked {int((~k).sum())} | notes {len(notes)}')
print(f'one surface {int((T.n_surfaces == 1).sum())} | merged (several surfaces in one component) {int((T.n_surfaces > 1).sum())} | conflicting scores between biopsy sessions {int(T.score_conflict.sum())}')
one = T[T.n_surfaces == 1]
rel = one.stl_vol_cc / one.roi_vol_cc
print(f'surface volume / spreadsheet ROI volume (single-surface targets): median {rel.median():.4f}, within 1% in {100 * rel.between(0.99, 1.01).mean():.1f}%, within 5% in {100 * rel.between(0.95, 1.05).mean():.1f}%')
print(f'vertex share in the assigned component: median {T.vertex_frac.median():.3f}, below 0.9 in {int((T.vertex_frac < 0.9).sum())}')
print('target score above examination score:', int((T.target_score > T.exam_score).sum()), '| examinations whose highest linked target score differs from the examination score:',
      int((T[k].groupby('Session_ID').target_score.max() != T[k].groupby('Session_ID').exam_score.first()).sum()))
print(pd.crosstab(T.exam_score, T.target_score.fillna(-9).astype(int)).to_string())
