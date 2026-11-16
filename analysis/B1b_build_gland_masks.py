# Step 3.3 (input part). Prostate (gland) masks on the grid of the PI-CAI-preprocessed T2 built by B1; ProViCNet needs them
# (it z-scores each sequence inside the gland). Nearest-neighbour resampling of the session prostate mask.
# Run from the repository root after B1 has finished:
#   python analysis/B1b_build_gland_masks.py
import os, glob
import numpy as np, SimpleITK as sitk

ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
UCLA = os.path.join(ROOT, 'MRI_US_Biopsy', 'Script')
IN = os.path.join(UCLA, 'nnUNet_raw_official', 'imagesTs')
OUT = os.path.join(UCLA, 'nnUNet_raw_official', 'masks')
SESS = os.path.join(UCLA, 'output', 'standardized_ucla_sessions_new')
os.makedirs(OUT, exist_ok=True)
# Pass 1: store images as rounded int16 (PI-CAI archives are integer; files become ~3x smaller to upload). Idempotent.
conv = 0
for f in sorted(glob.glob(os.path.join(IN, 'UCLA_*_000[012].nii.gz'))):
    r = sitk.ImageFileReader()
    r.SetFileName(f)
    r.ReadImageInformation()
    if r.GetPixelID() != sitk.sitkInt16:
        sitk.WriteImage(sitk.Cast(sitk.Round(sitk.ReadImage(f, sitk.sitkFloat32)), sitk.sitkInt16), f, True)
        conv += 1
print('images converted to int16:', conv)
n = bad = 0
for f in sorted(glob.glob(os.path.join(IN, 'UCLA_*_0000.nii.gz'))):
    sid = os.path.basename(f)[5:-12]
    o = os.path.join(OUT, f'UCLA_{sid}_mask.nii.gz')
    if os.path.exists(o):
        n += 1
        continue
    t2 = sitk.ReadImage(f)
    pm = sitk.ReadImage(os.path.join(SESS, sid, 'prostate_mask.nii.gz'))
    m = sitk.Resample(sitk.Cast(pm > 0, sitk.sitkUInt8), t2, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    vox = int(sitk.GetArrayViewFromImage(m).sum())
    vol_cc = vox * np.prod(t2.GetSpacing()) / 1000.0
    if vox == 0:
        bad += 1
        print('EMPTY MASK', sid)
        continue
    sitk.WriteImage(m, o, True)
    n += 1
    if n <= 3:
        print(sid, 'mask volume cc', round(vol_cc, 1), 'size', m.GetSize())
print('masks written/present', n, '| empty', bad)
