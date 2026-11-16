# PROSTATEx replication, step 1: match PROSTATEx training patients to their PI-CAI copies by transverse T2 geometry (headers only).
# The WorldMatrix of the PROSTATEx images table places its origin at the CORNER of voxel (0,0,0); image files (ITK, DICOM
# ImagePositionPatient) place it at the voxel CENTRE. The two differ by exactly half a voxel along each axis (0.25, 0.25 and 1.5 mm for
# 0.5 x 0.5 x 3.0 mm voxels = 1.54 mm), which is more than the 1.0 mm tolerance. The origin is therefore converted to the voxel centre first:
#   origin_centre = WorldMatrix @ (0.5, 0.5, 0.5, 1).   Match rule: identical matrix size, spacing difference
#   < 0.01 mm, origin difference < 1.0 mm, unique match. Checks: matched patients, and findings within 2 voxels of the ijk given in the
#   PROSTATEx images table, which uses the same corner convention (voxel index = floor(continuous corner index)).
# Image headers and finding positions only. An independent label check is printed at the end.
#   python analysis/R1b_match_prostatex_to_picai.py
import os
import numpy as np, pandas as pd
import SimpleITK as sitk

RAW = 'outputs/replication/raw/ProstateX-TrainingLesionInformationv2/'
OUT = 'outputs/replication/'
PICAI = os.environ.get('PROSTATE_DATA_ROOT', '../..') + '/PI-CAI_Validation/'
F = pd.read_csv(RAW + 'ProstateX-Findings-Train.csv')
I = pd.read_csv(RAW + 'ProstateX-Images-Train.csv')
ms = pd.read_csv(PICAI + 'picai_labels-main/clinical_information/marksheet.csv')
Hd = pd.read_csv(OUT + 'R1_picai_t2_headers.csv')                                  # PI-CAI RUMC T2 headers, written by R1
print(f'PI-CAI RUMC studies with a T2 file: {len(Hd)} | PROSTATEx training patients {F.ProxID.nunique()}, findings rows {len(F)}')

t2 = I[I.Name.str.contains('t2_tse_tra', case=False)].copy()
match, notes, dists = [], [], []
for pid, g in t2.groupby('ProxID'):
    cands = {}
    for _, r in g.drop_duplicates('DCMSerNum').iterrows():
        w = np.array(list(map(float, r.WorldMatrix.split(',')))).reshape(4, 4)
        sp = list(map(float, r.VoxelSpacing.split(',')))
        dim = list(map(int, r.Dim.split('x')))[:3]
        org = (w @ np.array([0.5, 0.5, 0.5, 1.0]))[:3]                             # corner -> centre of the first voxel
        d = np.sqrt((Hd.ox - org[0]) ** 2 + (Hd.oy - org[1]) ** 2 + (Hd.oz - org[2]) ** 2)
        ok = Hd[(Hd['size'] == ' '.join(map(str, dim))) & (abs(Hd.sx - sp[0]) < 0.01) & (abs(Hd.sy - sp[1]) < 0.01) & (abs(Hd.sz - sp[2]) < 0.01) & (d < 1.0)]
        for i, h in ok.iterrows():
            cands[(h.patient_id, h.study_id)] = (h.path, int(r.DCMSerNum), float(d[i]))
    if len(cands) == 1:
        (p, s), (path, ser, dd) = next(iter(cands.items()))
        match.append(dict(ProxID=pid, patient_id=int(p), study_id=int(s), t2_path=path, DCMSerNum=ser, origin_diff_mm=dd))
    else:
        notes.append(dict(ProxID=pid, n_matches=len(cands)))
M = pd.DataFrame(match)
dup = M.duplicated(['patient_id', 'study_id'], keep=False)
if dup.any():                                                                      # one PI-CAI study claimed by two PROSTATEx patients: not unique
    notes += [dict(ProxID=p, n_matches=-1) for p in M[dup].ProxID]
    M = M[~dup]
M.to_csv(OUT + 'match_table_R1b.csv', index=False)
pd.DataFrame(notes).to_csv(OUT + 'R1b_unmatched.csv', index=False)
n_pat = F.ProxID.nunique()
print(f'matched patients: {len(M)} of {n_pat}; '
      f'no match {sum(1 for x in notes if x["n_matches"] == 0)}, multiple {sum(1 for x in notes if x["n_matches"] > 1)}, shared study {sum(1 for x in notes if x["n_matches"] == -1)}')
print(f'origin difference after the half-voxel conversion: median {M.origin_diff_mm.median():.4f} mm, max {M.origin_diff_mm.max():.4f} mm')

# Check: finding position mapped into the PI-CAI T2 against the ijk of the PROSTATEx images table (matched T2 series)
rows = []
for _, m in M.iterrows():
    rd = sitk.ImageFileReader(); rd.SetFileName(m.t2_path); rd.ReadImageInformation()
    ref = sitk.Image([int(v) for v in rd.GetSize()], sitk.sitkUInt8)
    ref.SetOrigin(rd.GetOrigin()); ref.SetSpacing(rd.GetSpacing()); ref.SetDirection(rd.GetDirection())
    for _, f in F[F.ProxID == m.ProxID].iterrows():
        pos = [float(v) for v in f.pos.split()]
        idx = np.array(ref.TransformPhysicalPointToContinuousIndex(pos))           # centre convention
        r = I[(I.ProxID == m.ProxID) & (I.fid == f.fid) & (I.DCMSerNum == m.DCMSerNum) & I.Name.str.contains('t2_tse_tra', case=False)]
        r = r[r.pos.map(lambda p: np.allclose([float(v) for v in p.split()], pos, atol=1e-3))] if len(r) > 1 else r
        ijk = np.array(list(map(float, r.ijk.iloc[0].split()))) if len(r) else np.full(3, np.nan)
        inside = bool(np.all(idx >= -0.5) and np.all(idx <= np.array(rd.GetSize()) - 0.5))
        rows.append(dict(ProxID=m.ProxID, fid=f.fid, i=idx[0], j=idx[1], k=idx[2], ijk_i=ijk[0], ijk_j=ijk[1], ijk_k=ijk[2], inside_t2=int(inside),
                         maxdiff=float(np.nanmax(np.abs(idx - ijk))) if np.isfinite(ijk).all() else np.nan))
C = pd.DataFrame(rows)
C.to_csv(OUT + 'R1b_coordinate_check.csv', index=False)
ok, tot = int((C.maxdiff <= 2).sum()), int(C.maxdiff.notna().sum())
print(f'findings within two voxels: {ok} of {tot} ({100 * ok / max(tot, 1):.1f}%); '
      f'without ijk {int(C.maxdiff.isna().sum())}; median max difference {C.maxdiff.median():.2f} voxels; findings inside the T2 volume {int(C.inside_t2.sum())} of {len(C)}')
print(f'findings rows of matched patients: {len(C)} | duplicated (ProxID, fid) rows: {int(F[F.ProxID.isin(M.ProxID)].duplicated(["ProxID", "fid"]).sum())}')

# Independent check of the match: the PI-CAI case label against the PROSTATEx finding labels of the matched patient.
lab = F.groupby('ProxID').ClinSig.any().rename('px_any_clinsig').reset_index().merge(M, on='ProxID').merge(ms[['patient_id', 'study_id', 'case_csPCa', 'center']], on=['patient_id', 'study_id'])
print('matched studies by PI-CAI centre:', lab.center.value_counts().to_dict())
print('PROSTATEx "any clinically significant finding" (rows) against PI-CAI case_csPCa (columns):')
print(pd.crosstab(lab.px_any_clinsig, lab.case_csPCa).to_string())
print('GATES', 'PASSED' if (g1 and g2) else 'FAILED')
