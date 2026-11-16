# Converts a model's probability volumes to the session 1 mm grid in the SegResNet npz format
#   <Session_ID>.npz : heatmap (x,y,z float32, zero outside the prostate dilated by 4 voxels), affine (RAS 4x4), pros_mask
# so that B5_extract_candidates_generic.py can read them.
#
# Two input kinds:
#   --kind nnunet : folder with UCLA_<sid>.npz (key softmax, class 1, z,y,x) + UCLA_<sid>.pkl, geometry taken from --inputs/UCLA_<sid>_0000.nii.gz
#   --kind nifti  : folder with <prefix><sid>.nii.gz probability volumes that carry their own geometry (e.g. ProViCNet)
# Run from the repository root:
#   python analysis/B3_probmaps_to_session_grid.py --kind nnunet --pred <folder> --inputs <imagesTs folder> --out results/nnunet_official_probmaps
import os, argparse, pickle, warnings
import numpy as np, pandas as pd, SimpleITK as sitk, nibabel as nib
from scipy.ndimage import binary_dilation
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument('--kind', required=True, choices=['nnunet', 'nifti'])
ap.add_argument('--pred', required=True)
ap.add_argument('--inputs')
ap.add_argument('--prefix', default='UCLA_')
ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
SESS = os.path.join(ROOT, 'MRI_US_Biopsy', 'Script', 'output', 'standardized_ucla_sessions_new')
os.makedirs(a.out, exist_ok=True)
ep = pd.read_csv('results/tables/EXAMINATION_LEVEL_ENDPOINTS.csv')
excl = set(pd.read_csv('outputs/EXCLUDED_SESSIONS.csv').Session_ID)
todo = [s for s in ep.Session_ID if s not in excl]
if a.limit:
    todo = todo[:a.limit]
done = missing = 0
for sid in todo:
    o = os.path.join(a.out, sid + '.npz')
    if os.path.exists(o):
        done += 1
        continue
    if a.kind == 'nnunet':
        f = os.path.join(a.pred, f'{a.prefix}{sid}.npz')
        if not os.path.exists(f):
            missing += 1
            continue
        ref = sitk.ReadImage(os.path.join(a.inputs, f'{a.prefix}{sid}_0000.nii.gz'))
        sm = np.load(f)['softmax'][1].astype(np.float32)                     # (z,y,x)
        full = tuple(ref.GetSize()[::-1])
        if sm.shape != full:                                                 # nnU-Net cropped to the non-zero region
            meta = pickle.load(open(f[:-4] + '.pkl', 'rb'))
            bb = meta.get('crop_bbox')
            assert bb is not None, f'{sid}: shape {sm.shape} != {full} and no crop_bbox'
            tmp = np.zeros(full, np.float32)
            tmp[bb[0][0]:bb[0][1], bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]] = sm
            sm = tmp
        prob = sitk.GetImageFromArray(sm)
        prob.CopyInformation(ref)
    else:
        f = os.path.join(a.pred, f'{a.prefix}{sid}.nii.gz')
        if not os.path.exists(f):
            missing += 1
            continue
        prob = sitk.Cast(sitk.ReadImage(f), sitk.sitkFloat32)
    pm_path = os.path.join(SESS, sid, 'prostate_mask.nii.gz')
    pm = sitk.ReadImage(pm_path)
    res = sitk.Resample(prob, pm, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)   # physical space, LPS, onto the session grid
    heat = np.clip(sitk.GetArrayFromImage(res).transpose(2, 1, 0), 0, 1)     # -> (x,y,z) as script 04 expects
    nii = nib.load(pm_path)
    mask = np.asarray(nii.dataobj) > 0
    assert mask.shape == heat.shape, f'{sid}: {mask.shape} vs {heat.shape}'
    heat[~binary_dilation(mask, iterations=4)] = 0.0                          # same masking as script 04, line 311
    np.savez_compressed(o, heatmap=heat.astype(np.float32), affine=nii.affine.astype(np.float64), pros_mask=mask)
    done += 1
print(f'written/present {done} | predictions missing {missing} | of {len(todo)} -> {a.out}')
