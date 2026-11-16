# PROSTATEx replication, step 2: model inputs for the matched PROSTATEx patients, prepared exactly as for the UCLA cohort (script B1):
#   T2 resampled to 0.5 x 0.5 x 3.0 mm; crop of 256 x 256 x max(20, ceil((gland extent + 30 mm) / 3)) centred on the gland;
#   ADC and high-b-value image resampled onto the T2 grid; integer storage; gland mask on the same grid (script B1b).
# Images: PI-CAI public copies of the same examinations (t2w, adc, hbv). Gland mask: public PI-CAI whole-gland delineations (Bosma22b).
# Also written: the 1-mm grid and gland mask used for candidate extraction (script 01, resample_image_to_isotropic) and the input
# content check of script B7 with unchanged thresholds. Outcome-blind (no finding label is read).
#   python analysis/R2_build_prostatex_inputs.py
import os, time, warnings
import numpy as np, pandas as pd, SimpleITK as sitk
warnings.filterwarnings("ignore")
from picai_prep.preprocessing import resample_img, resample_to_reference_scan

REP = 'outputs/replication/'
PICAI = os.environ.get('PROSTATE_DATA_ROOT', '../..') + '/PI-CAI_Validation/'
GL = PICAI + 'picai_labels-main/anatomical_delineations/whole_gland/AI/Bosma22b/'
ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
OUTD = os.path.join(ROOT, 'PROSTATEx_replication')
for d in ('imagesTs', 'masks', 'grid1mm'):
    os.makedirs(os.path.join(OUTD, d), exist_ok=True)
M = pd.read_csv(REP + 'match_table_R1b.csv')


def iso1mm(image, ref=None, label=False):
    """pipeline/01 resample_image_to_isotropic: 1 mm grid over the full T2 field of view."""
    rs = sitk.ResampleImageFilter()
    if ref is not None:
        rs.SetReferenceImage(ref)
    else:
        size = [int(v) for v in np.ceil(np.array(image.GetSize(), dtype=np.int32) * np.array(image.GetSpacing()))]
        rs.SetOutputSpacing((1.0, 1.0, 1.0)); rs.SetSize(size); rs.SetOutputDirection(image.GetDirection()); rs.SetOutputOrigin(image.GetOrigin())
    rs.SetInterpolator(sitk.sitkNearestNeighbor if label else sitk.sitkBSpline)
    return rs.Execute(image)


log, t0 = [], time.time()
for n, m in M.iterrows():
    c = m.ProxID
    out = [os.path.join(OUTD, 'imagesTs', f'PX_{c}_000{k}.nii.gz') for k in range(3)]
    mo, go = os.path.join(OUTD, 'masks', f'PX_{c}_mask.nii.gz'), os.path.join(OUTD, 'grid1mm', f'{c}_prostate_mask.nii.gz')
    r = dict(ProxID=c, patient_id=m.patient_id, study_id=m.study_id, ok=0)
    try:
        base = m.t2_path[:-len('_t2w.mha')]
        t2, adc, dwi = (sitk.Cast(sitk.ReadImage(base + s), sitk.sitkFloat32) for s in ('_t2w.mha', '_adc.mha', '_hbv.mha'))
        pm = sitk.ReadImage(GL + f'{m.patient_id}_{m.study_id}.nii.gz')
        same = pm.GetSize() == t2.GetSize() and np.allclose(pm.GetOrigin(), t2.GetOrigin(), atol=1e-3) and np.allclose(pm.GetSpacing(), t2.GetSpacing(), atol=1e-4)
        pm = sitk.Cast(pm > 0, sitk.sitkUInt8)
        if not same:
            pm = sitk.Resample(pm, t2, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        st = sitk.LabelShapeStatisticsImageFilter(); st.Execute(pm)
        if not st.HasLabel(1):
            raise ValueError('empty gland mask')
        t2r = resample_img(t2, out_spacing=[3.0, 0.5, 0.5])
        cen = np.array(t2r.TransformPhysicalPointToContinuousIndex(st.GetCentroid(1)))
        bb = st.GetBoundingBox(1)
        ext_mm = np.array(bb[3:]) * np.array(pm.GetSpacing())
        want = [256, 256, max(20, int(np.ceil((ext_mm[2] + 30.0) / 3.0)))]
        size = np.array(t2r.GetSize())
        start, csize = [], []
        for d in range(3):
            w = min(want[d], int(size[d]))
            s0 = max(0, min(int(round(cen[d] - w / 2.0)), int(size[d]) - w))
            start.append(s0); csize.append(w)
        t2c = sitk.RegionOfInterest(t2r, csize, start)
        adcc = resample_to_reference_scan(adc, reference_scan_original=t2c)
        dwic = resample_to_reference_scan(dwi, reference_scan_original=t2c)
        for img, o in zip((t2c, adcc, dwic), out):
            sitk.WriteImage(sitk.Cast(sitk.Round(img), sitk.sitkInt16), o, True)
        gm = sitk.Resample(pm, t2c, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        sitk.WriteImage(gm, mo, True)
        g1 = iso1mm(pm, ref=iso1mm(t2), label=True)                                # 1-mm grid for candidate extraction
        sitk.WriteImage(g1, go, True)
        mk = sitk.GetArrayFromImage(gm) == 1
        r.update(ok=1, mask_same_grid=int(same), size=str(tuple(csize)), gland_cc=float(mk.sum() * 0.5 * 0.5 * 3.0 / 1000.0), gland_cc_1mm=float(sitk.GetArrayViewFromImage(g1).sum() / 1000.0),
                 gland_inside=int(all(0 <= cc - s < w for cc, s, w in zip(cen, start, csize))), t2_native=str(tuple(round(v, 2) for v in t2.GetSpacing())), gland_vox=int(mk.sum()))
        for k, nm in ((0, 't2'), (1, 'adc'), (2, 'hbv')):                          # content check, as script B7
            a = sitk.GetArrayFromImage(sitk.ReadImage(out[k])).astype(np.float32)[mk]
            vals, cnt = np.unique(a, return_counts=True)
            nz = a[a != 0]
            r[nm + '_zero_frac'] = float((a == 0).mean())
            r[nm + '_mode_frac_nonzero'] = float(cnt[vals != 0].max() / max((a != 0).sum(), 1)) if (a != 0).any() else 1.0
            r[nm + '_cv_nonzero'] = float(nz.std() / (abs(nz.mean()) + 1e-6)) if len(nz) else 0.0
    except Exception as e:
        r['error'] = str(e)[:200]
    log.append(r)
    if n % 40 == 0:
        print(n, c, f'{(time.time() - t0) / 60:.1f} min', flush=True)
Q = pd.DataFrame(log)
k = Q.ok == 1
Q.loc[k, 'flag_coverage'] = ((Q.adc_zero_frac >= 0.20) | (Q.hbv_zero_frac >= 0.20) | (Q.t2_zero_frac >= 0.20))[k].astype(int)
Q.loc[k, 'flag_saturated_hbv'] = ((Q.hbv_mode_frac_nonzero >= 0.50) | (Q.hbv_cv_nonzero < 0.05))[k].astype(int)
Q.loc[k, 'flag_saturated_adc'] = ((Q.adc_mode_frac_nonzero >= 0.50) | (Q.adc_cv_nonzero < 0.05))[k].astype(int)
Q.loc[k, 'qc_fail'] = ((Q.flag_coverage + Q.flag_saturated_hbv + Q.flag_saturated_adc) > 0)[k].astype(int)
Q.to_csv(REP + 'R2_inputs_log.csv', index=False)
print(f'patients {len(Q)} | prepared {int(k.sum())} | failed {int((~k).sum())} | gland mask on the T2 grid {int(Q.mask_same_grid.sum())} | gland inside crop {int(Q.gland_inside.sum())}')
print(f'content rule (B7): coverage {int(Q.flag_coverage.sum())} | saturated high-b {int(Q.flag_saturated_hbv.sum())} | saturated ADC {int(Q.flag_saturated_adc.sum())} | any {int(Q.qc_fail.sum())}')
print(Q[k][['size', 't2_native']].value_counts().head(6).to_string())
print('gland volume (mL): median %.1f, min %.1f, max %.1f' % (Q.gland_cc.median(), Q.gland_cc.min(), Q.gland_cc.max()))
if (~k).any():
    print(Q[~k][['ProxID', 'error']].to_string(index=False))
