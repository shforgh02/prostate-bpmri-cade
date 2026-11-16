# PROSTATEx replication, step 0: writes the PI-CAI T2 header table (R1_picai_t2_headers.csv) that R1b uses, and a first match of PROSTATEx
# training patients to their PI-CAI copies with the image origin as given in the PROSTATEx images table (corner convention). This match
# fails because of the half-voxel origin difference; R1b repeats it with the origin converted to the voxel center.
# Match rule: identical matrix size, spacing difference < 0.01 mm, origin difference < 1.0 mm, unique match.
#   python analysis/R1_match_prostatex_to_picai.py
import glob, os, re, sys
import numpy as np, pandas as pd
import SimpleITK as sitk

RAW = 'outputs/replication/raw/ProstateX-TrainingLesionInformationv2/'
OUT = 'outputs/replication/'
PICAI = os.environ.get('PROSTATE_DATA_ROOT', '../..') + '/PI-CAI_Validation/'
F = pd.read_csv(RAW + 'ProstateX-Findings-Train.csv')
I = pd.read_csv(RAW + 'ProstateX-Images-Train.csv')
ms = pd.read_csv(PICAI + 'picai_labels-main/clinical_information/marksheet.csv')
rumc = ms[ms.center == 'RUMC']

# PI-CAI T2 headers (RUMC only), cached
hf = OUT + 'R1_picai_t2_headers.csv'
if not os.path.exists(hf):
    rows = []
    for _, r in rumc.iterrows():
        f = glob.glob(PICAI + f'picai_public_images_fold*/{r.patient_id}/{r.patient_id}_{r.study_id}_t2w.mha')
        if not f:
            continue
        rd = sitk.ImageFileReader(); rd.SetFileName(f[0]); rd.ReadImageInformation()
        rows.append(dict(patient_id=r.patient_id, study_id=r.study_id, path=f[0], size=' '.join(map(str, rd.GetSize())),
                         sx=rd.GetSpacing()[0], sy=rd.GetSpacing()[1], sz=rd.GetSpacing()[2], ox=rd.GetOrigin()[0], oy=rd.GetOrigin()[1], oz=rd.GetOrigin()[2]))
    pd.DataFrame(rows).to_csv(hf, index=False)
Hd = pd.read_csv(hf)
print(f'PI-CAI RUMC studies with a T2 file: {len(Hd)}')

t2 = I[I.Name.str.contains('t2_tse_tra', case=False)].copy()
def geom(r):
    w = np.array(list(map(float, r.WorldMatrix.split(',')))).reshape(4, 4)
    sp = list(map(float, r.VoxelSpacing.split(',')))
    dim = list(map(int, r.Dim.split('x')))[:3]
    return dim, sp, w[:3, 3]
match, notes = [], []
for pid, g in t2.groupby('ProxID'):
    cands = set()
    for _, r in g.drop_duplicates('DCMSerNum').iterrows():
        dim, sp, org = geom(r)
        ok = Hd[(Hd['size'] == ' '.join(map(str, dim))) & (abs(Hd.sx - sp[0]) < 0.01) & (abs(Hd.sy - sp[1]) < 0.01) & (abs(Hd.sz - sp[2]) < 0.01) &
                (np.sqrt((Hd.ox - org[0]) ** 2 + (Hd.oy - org[1]) ** 2 + (Hd.oz - org[2]) ** 2) < 1.0)]
        for _, h in ok.iterrows():
            cands.add((h.patient_id, h.study_id, h.path, int(r.DCMSerNum)))
    keys = {(c[0], c[1]) for c in cands}
    if len(keys) == 1:
        c = sorted(cands)[0]
        match.append(dict(ProxID=pid, patient_id=c[0], study_id=c[1], t2_path=c[2], DCMSerNum=c[3]))
    else:
        notes.append(dict(ProxID=pid, n_matches=len(keys)))
M = pd.DataFrame(match)
M.to_csv(OUT + 'match_table.csv', index=False)
pd.DataFrame(notes).to_csv(OUT + 'R1_unmatched.csv', index=False)
n_pat = F.ProxID.nunique()
print(f'matched patients: {len(M)} of {n_pat}; '
      f'no match {sum(1 for x in notes if x["n_matches"] == 0)}, multiple {sum(1 for x in notes if x["n_matches"] > 1)}')

# Coordinate check against the ijk of the matched T2 series
rows = []
for _, m in M.iterrows():
    img = sitk.ImageFileReader(); img.SetFileName(m.t2_path); img.ReadImageInformation()
    ref = sitk.Image([int(x) for x in img.GetSize()], sitk.sitkUInt8)
    ref.SetOrigin(img.GetOrigin()); ref.SetSpacing(img.GetSpacing()); ref.SetDirection(img.GetDirection())
    for _, f in F[F.ProxID == m.ProxID].iterrows():
        pos = [float(x) for x in f.pos.split()]
        idx = np.array(ref.TransformPhysicalPointToContinuousIndex(pos))
        r = I[(I.ProxID == m.ProxID) & (I.fid == f.fid) & (I.DCMSerNum == m.DCMSerNum) & I.Name.str.contains('t2_tse_tra', case=False)]
        ijk = np.array(list(map(float, r.ijk.iloc[0].split()))) if len(r) else np.full(3, np.nan)
        rows.append(dict(ProxID=m.ProxID, fid=f.fid, i=idx[0], j=idx[1], k=idx[2], ijk_i=ijk[0], ijk_j=ijk[1], ijk_k=ijk[2],
                         maxdiff=float(np.nanmax(np.abs(idx - ijk))) if np.isfinite(ijk).all() else np.nan))
C = pd.DataFrame(rows)
C.to_csv(OUT + 'R1_coordinate_check.csv', index=False)
ok = (C.maxdiff <= 2).sum()
tot = C.maxdiff.notna().sum()
print(f'findings within two voxels: {ok} of {tot} ({100 * ok / max(tot, 1):.1f}%); '
      f'findings without ijk {int(C.maxdiff.isna().sum())}; median max difference {C.maxdiff.median():.2f} voxels')
print(f'findings of matched patients: {len(C)}')
