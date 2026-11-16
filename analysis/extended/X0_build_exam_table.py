# Builds the per-examination table used by X1-X6.
#  - clinical: age/weight/field/vendor from the DICOM header of the analysed T2 series, PSA, volume, PSA density, suspicion score, grade,
#    numbers of cores, targets, target volume, radiologist distance (centre and surface) and recorded targeted yield
#  - per model and top-k (k = 1, 2, 3, 5): distance of the nearest candidate centre to a csPCa-positive core segment, and the
#    between-patient permutation probability of a hit at tolerances 2.5-15 mm (same chance model as A1)
#   python analysis/extended/X0_build_exam_table.py --exclude outputs/EXCLUDED_SESSIONS_v2.csv \
#          --candidates outputs/CANDIDATES_3MODELS.csv --topk 5 --tag ext
import os, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, 'analysis')
from _common import SD, CC, args, load, p2s, bbox

R = 'outputs/extended/'
os.makedirs(R, exist_ok=True)
a = args()
ep, cand, cores, ms, models = load(a)                       # candidates up to rank a.topk (5)
TOLS = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
KS = (1, 2, 3, 5)

# ---- DICOM header of the analysed T2 series (age is not in the clinical spreadsheets)
hdr_f = R + 'X0_dicom_header.csv'
if not os.path.exists(hdr_f):
    import pydicom
    man = pd.read_csv('results/tables/ANALYSIS_MANIFEST.csv').set_index('Session_ID')
    root = os.environ.get('PROSTATE_DATA_ROOT', '../..') + '/MRI_US_Biopsy/manifest-1694710246744/'
    md = pd.read_csv(root + 'metadata.csv').set_index('Series UID')
    rows = []
    for s in ep.Session_ID:
        r = dict(Session_ID=s)
        try:
            loc = root + md.loc[man.loc[s, 'MRI_Series_UID'], 'File Location'].replace('\\', '/').lstrip('./')
            f = sorted(glob.glob(loc + '/*.dcm'))[0]
            d = pydicom.dcmread(f, stop_before_pixels=True)
            age = str(getattr(d, 'PatientAge', '') or '')
            r.update(age=float(age[:-1]) if age.endswith('Y') else np.nan, weight=float(getattr(d, 'PatientWeight', 0) or 0) or np.nan,
                     field=float(getattr(d, 'MagneticFieldStrength', 0) or 0) or np.nan, vendor=str(getattr(d, 'Manufacturer', '')),
                     study_desc=str(getattr(d, 'StudyDescription', '')))
        except Exception as ex:
            r.update(error=str(ex)[:80])
        rows.append(r)
    pd.DataFrame(rows).to_csv(hdr_f, index=False)
H = pd.read_csv(hdr_f).set_index('Session_ID')

# ---- examination table
pos = cores[cores.Is_Valid_csPCa_Trajectory == 1]
SEG = {s: (g[CC[:3]].to_numpy(float), g[CC[3:]].to_numpy(float)) for s, g in pos.groupby('Session_ID_Matched')}
cc = cores.groupby('Session_ID_Matched').agg(n_cores=('Is_csPCa', 'size'), n_targeted=('Is_Targeted_Core', 'sum'), n_pos_cores=('Is_csPCa', 'sum'))
T = ep[['Session_ID', 'Patient_ID', 'Exact_Exam_UCLA_Score', 'Is_csPCa', 'Has_Valid_3D_Trajectory', 'Targeted_Biopsy_csPCa_Yield_C',
        'Systematic_Biopsy_csPCa_Hit', 'Has_Human_Target_Contour', 'Human_Target_CoM_Dist_csPCa_mm']].copy()
T = T.rename(columns={'Exact_Exam_UCLA_Score': 'score', 'Is_csPCa': 'cspca', 'Targeted_Biopsy_csPCa_Yield_C': 'C', 'Human_Target_CoM_Dist_csPCa_mm': 'rad_dist'})
m = ms.loc[T.Session_ID]
T['psa'] = m.PSA.to_numpy(float)
T['vol'] = m.MRI_Prostate_Vol_cc.to_numpy(float)
T['psad'] = np.where(T.vol > 0, T.psa / T.vol, np.nan)
T['grade'] = m.Sess_Max_ISUP.to_numpy()
T['n_targets'] = m.Human_Rad_Mask_Num_Components.to_numpy()
T['target_vol'] = m.Human_Rad_Mask_Vol_cc.to_numpy(float)
T['rad_surf_dist'] = m.Human_Sess_Surface3D_Dist_csPCa_mm.to_numpy(float)
T['rad_puncture'] = m.Human_Sess_DirectPuncture_0mm_csPCa.to_numpy()
T = T.join(cc, on='Session_ID').join(H[['age', 'weight', 'field', 'vendor']], on='Session_ID')
T['evaluable'] = ((T.cspca == 1) & (T.Has_Valid_3D_Trajectory == 1)).astype(int)
for mod in models:
    T[f'maxprob_{mod}'] = T.Session_ID.map(cand[cand.Model == mod].groupby('Session_ID').Max_Prob.max()).fillna(0).to_numpy()

# ---- per model, top-k: observed nearest distance and permutation hit probabilities (evaluable cancers only)
BB = {s: bbox(ms, s) for s in T.Session_ID}
EV = T.loc[T.evaluable == 1, 'Session_ID'].tolist()
P = []
for mod in models:
    cm = cand[cand.Model == mod]
    for k in KS:
        C = {s: g.sort_values('Candidate_Rank')[['CoM_LPS_X_mm', 'CoM_LPS_Y_mm', 'CoM_LPS_Z_mm']].to_numpy(float)
             for s, g in cm[cm.Candidate_Rank <= k].groupby('Session_ID')}
        NORM = {s: (C[s] - BB[s][0]) / BB[s][1] for s in C}
        for s in EV:
            A_, B_ = SEG[s]
            mn, sp = BB[s]
            n = len(C.get(s, []))
            own = p2s(C[s], A_, B_).min() if n else np.inf
            donors = [v for j, v in NORM.items() if j != s and len(v) == n] if n else []
            dmin = np.array([p2s(mn + v * sp, A_, B_).min() for v in donors]) if donors else np.array([np.inf])
            r = dict(Session_ID=s, model=mod, k=k, n_cand=n, dist=own)
            for t in TOLS:
                r[f'perm_{t}'] = float((dmin <= t).mean())
            P.append(r)
        print(mod, 'k =', k, 'done', flush=True)
pd.DataFrame(P).to_csv(R + 'X0_perm_by_k.csv', index=False)
T.to_csv(R + 'X0_exam_table.csv', index=False)
print('exams', len(T), '| evaluable', len(EV), '| age available', int(T.age.notna().sum()), '| median age', T.age.median())
