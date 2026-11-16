# Builds nnU-Net / ProViCNet inputs for the analysed UCLA exams with PI-CAI preprocessing:
#   raw intensities; T2 resampled to 0.5 x 0.5 x 3.0 mm; crop centred on the prostate; ADC and high-b resampled onto the T2 grid.
# Series selection is copied from pipeline/01_phase1_preprocessing_and_standardization.py (standardize_ucla_cohort) so the SAME series are used.
# Deviation from PI-CAI: the crop is centred on the prostate mask (not the image centre) and grows in z for large glands.
# Resumable (skips finished cases). Run from the repository root:
#   python analysis/B1_build_native_inputs.py [--limit N]
import os, sys, argparse, time, warnings
import numpy as np, pandas as pd, SimpleITK as sitk
warnings.filterwarnings("ignore")
from picai_prep.preprocessing import resample_img, resample_to_reference_scan

ap = argparse.ArgumentParser()
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))                     # data root
UCLA = os.path.join(ROOT, 'MRI_US_Biopsy')
MANI = os.path.join(UCLA, 'manifest-1694710246744')
SESS = os.path.join(UCLA, 'Script', 'output', 'standardized_ucla_sessions_new')   # holds prostate_mask.nii.gz per session
OUTD = os.path.join(UCLA, 'Script', 'nnUNet_raw_official', 'imagesTs')
os.makedirs(OUTD, exist_ok=True)
ep = pd.read_csv('results/tables/EXAMINATION_LEVEL_ENDPOINTS.csv')
excl = set(pd.read_csv('outputs/EXCLUDED_SESSIONS.csv').Session_ID)
audit = pd.read_csv('outputs/highb_series_audit.csv', low_memory=False).set_index('session')
todo = [s for s in ep.Session_ID if s not in excl]
if a.limit:
    todo = todo[:a.limit]
meta = pd.read_csv(os.path.join(MANI, 'metadata.csv'))
meta.columns = [c.strip() for c in meta.columns]
bx = pd.read_excel(os.path.join(UCLA, 'TCIA-Biopsy-Data_2020-07-14.xlsx')).dropna(subset=['Bx Tip X (MRI Coord)'])
UID = {}
for pid, uid in zip(bx['Patient Number'].astype(str).str.strip(), bx['Series Instance UID (MRI)'].astype(str).str.strip()):
    if uid and uid != 'nan':
        UID.setdefault(f'{pid}_{uid[-5:]}', uid)


def read_series(folder, uid):
    r = sitk.ImageSeriesReader()
    names = r.GetGDCMSeriesFileNames(folder, uid)
    if not names:
        raise ValueError('no DICOM for series ' + uid)
    r.SetFileNames(names)
    return r.Execute()


def loc(row):
    return os.path.join(MANI, row['File Location'].lstrip('.\\'))


def pick(study):
    """Same rules as script 01: first ADC series; diffusion series by name pattern order."""
    adc = study[study['Series Description'].str.contains('ADC|Apparent', case=False, na=False)]
    dwi = study[~study['Series Description'].str.contains('ADC|Apparent|Localizer|Screen|T2|TSE|SPC|AX|Sag|Cor|Survey|Recon', case=False, na=False)]
    if adc.empty or dwi.empty:
        return None, None
    row = None
    for pat in ['1400', '1500', 'CALCBVAL', 'TRACEW', 'DWI', 'diff']:
        m = dwi[dwi['Series Description'].str.contains(pat, case=False, na=False)]
        if not m.empty:
            row = m.iloc[0]
            break
    return adc.iloc[0], (row if row is not None else dwi.iloc[0])


log = []
t0 = time.time()
for n, sid in enumerate(todo):
    out = [os.path.join(OUTD, f'UCLA_{sid}_000{c}.nii.gz') for c in range(3)]
    if all(os.path.exists(o) for o in out):
        continue
    try:
        t2row = meta[meta['Series UID'] == UID[sid]].iloc[0]
        adcrow, dwirow = pick(meta[meta['Study UID'] == t2row['Study UID']])
        desc_ok = str(dwirow['Series Description']).strip() == str(audit.loc[sid, 'dwi_desc']).strip()
        t2 = sitk.Cast(read_series(loc(t2row), t2row['Series UID']), sitk.sitkFloat32)
        adc = sitk.Cast(read_series(loc(adcrow), adcrow['Series UID']), sitk.sitkFloat32)
        dwi = sitk.Cast(read_series(loc(dwirow), dwirow['Series UID']), sitk.sitkFloat32)
        t2r = resample_img(t2, out_spacing=[3.0, 0.5, 0.5])
        # crop centred on the prostate (mask is in the same physical frame as the T2)
        pm = sitk.ReadImage(os.path.join(SESS, sid, 'prostate_mask.nii.gz'))
        st = sitk.LabelShapeStatisticsImageFilter()
        st.Execute(sitk.Cast(pm > 0, sitk.sitkUInt8))
        cen = np.array(t2r.TransformPhysicalPointToContinuousIndex(st.GetCentroid(1)))
        bb = st.GetBoundingBox(1)                                                 # x0,y0,z0,sx,sy,sz in mask voxels
        ext_mm = np.array(bb[3:]) * np.array(pm.GetSpacing())                     # mask axes follow the T2 axes
        want = [256, 256, max(20, int(np.ceil((ext_mm[2] + 30.0) / 3.0)))]
        size = np.array(t2r.GetSize())
        start, csize = [], []
        for d in range(3):
            w = min(want[d], int(size[d]))
            s0 = int(round(cen[d] - w / 2.0))
            s0 = max(0, min(s0, int(size[d]) - w))
            start.append(s0)
            csize.append(w)
        t2c = sitk.RegionOfInterest(t2r, csize, start)
        adcc = resample_to_reference_scan(adc, reference_scan_original=t2c)
        dwic = resample_to_reference_scan(dwi, reference_scan_original=t2c)
        inside = all(0 <= c - s < w for c, s, w in zip(cen, start, csize))
        for img, o in zip((t2c, adcc, dwic), out):
            sitk.WriteImage(sitk.Cast(sitk.Round(img), sitk.sitkInt16), o, True)   # integer storage like PI-CAI archives; ~3x smaller uploads
        aa = sitk.GetArrayViewFromImage(adcc)
        log.append(dict(Session_ID=sid, ok=1, same_dwi_series_as_before=int(desc_ok), size=str(tuple(csize)), prostate_centre_inside=int(inside),
                        prostate_z_mm=round(float(ext_mm[2]), 1), t2_native_spacing=str(tuple(round(x, 2) for x in t2.GetSpacing())), adc_p50=float(np.percentile(aa, 50)), adc_p99=float(np.percentile(aa, 99))))
    except Exception as e:
        log.append(dict(Session_ID=sid, ok=0, error=str(e)[:200]))
    if n % 50 == 0:
        print(n, sid, f'{(time.time() - t0) / 60:.1f} min', flush=True)
L = pd.DataFrame(log)
lp = 'outputs/B1_build_log.csv'
if os.path.exists(lp) and len(L):
    L = pd.concat([pd.read_csv(lp), L]).drop_duplicates('Session_ID', keep='last')
if len(L):
    L.to_csv(lp, index=False)
    print('processed', len(L), '| ok', int(L.ok.sum()), '| same diffusion series as before', int(L.get('same_dwi_series_as_before', pd.Series(dtype=float)).sum()),
          '| prostate centre inside crop', int(L.get('prostate_centre_inside', pd.Series(dtype=float)).sum()))
    print(L[L.ok == 1][['size', 't2_native_spacing']].value_counts().head(8).to_string())
print('files in output folder:', len(os.listdir(OUTD)))
