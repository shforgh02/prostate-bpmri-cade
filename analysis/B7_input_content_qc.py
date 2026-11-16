# Content check of the model inputs inside the prostate, for every analysed exam (outcome-blind).
# Flags: diffusion not covering the gland (zero ADC or zero high-b in a large share of gland voxels) and
#        saturated / constant high-b image (most gland voxels at one value).
# Run from the repository root:  python analysis/B7_input_content_qc.py
import os, glob, numpy as np, pandas as pd, SimpleITK as sitk
ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
D = os.path.join(ROOT, 'MRI_US_Biopsy', 'Script', 'nnUNet_raw_official')
rows = []
for f in sorted(glob.glob(os.path.join(D, 'imagesTs', 'UCLA_*_0000.nii.gz'))):
    sid = os.path.basename(f)[5:-12]
    m = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(D, 'masks', f'UCLA_{sid}_mask.nii.gz'))) == 1
    r = dict(Session_ID=sid, gland_vox=int(m.sum()))
    for c, nm in ((0, 't2'), (1, 'adc'), (2, 'hbv')):
        a = sitk.GetArrayFromImage(sitk.ReadImage(f.replace('_0000', f'_000{c}'))).astype(np.float32)[m]
        vals, cnt = np.unique(a, return_counts=True)
        r[nm + '_zero_frac'] = float((a == 0).mean())
        r[nm + '_mode_frac_nonzero'] = float(cnt[vals != 0].max() / max((a != 0).sum(), 1)) if (a != 0).any() else 1.0
        r[nm + '_mean'] = float(a.mean()); r[nm + '_cv'] = float(a.std() / (abs(a.mean()) + 1e-6))
        nz = a[a != 0]
        r[nm + '_cv_nonzero'] = float(nz.std() / (abs(nz.mean()) + 1e-6)) if len(nz) else 0.0
    rows.append(r)
Q = pd.DataFrame(rows)
Q['flag_coverage'] = ((Q.adc_zero_frac >= 0.20) | (Q.hbv_zero_frac >= 0.20) | (Q.t2_zero_frac >= 0.20)).astype(int)
Q['flag_saturated_hbv'] = ((Q.hbv_mode_frac_nonzero >= 0.50) | (Q.hbv_cv_nonzero < 0.05)).astype(int)   # one value, or nearly constant (e.g. saturated at 4095)
Q['flag_saturated_adc'] = ((Q.adc_mode_frac_nonzero >= 0.50) | (Q.adc_cv_nonzero < 0.05)).astype(int)
Q['qc_fail'] = ((Q.flag_coverage + Q.flag_saturated_hbv + Q.flag_saturated_adc) > 0).astype(int)
Q.to_csv('outputs/B7_input_content_qc.csv', index=False)
print('exams', len(Q), '| coverage flag', int(Q.flag_coverage.sum()), '| saturated high-b', int(Q.flag_saturated_hbv.sum()), '| saturated ADC', int(Q.flag_saturated_adc.sum()), '| any', int(Q.qc_fail.sum()))
print('zero-fraction quantiles (95%, 99%, max): ADC', np.round(Q.adc_zero_frac.quantile([.95, .99, 1]).values, 3), '| high-b', np.round(Q.hbv_zero_frac.quantile([.95, .99, 1]).values, 3))
print('non-zero CV quantiles (min, 1%, 5%, 50%): high-b', np.round(Q.hbv_cv_nonzero.quantile([0, .01, .05, .5]).values, 3), '| ADC', np.round(Q.adc_cv_nonzero.quantile([0, .01, .05, .5]).values, 3))
print(Q[Q.qc_fail == 1][['Session_ID', 'adc_zero_frac', 'hbv_zero_frac', 'hbv_cv_nonzero', 'adc_cv_nonzero', 'hbv_mode_frac_nonzero', 'adc_mode_frac_nonzero', 'hbv_mean']].round(2).to_string(index=False))
