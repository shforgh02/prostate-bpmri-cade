# PROSTATEx replication, step 5: agreement analysis.
#   Unit: PROSTATEx training finding (radiologist-identified, sampled by MR-guided biopsy) of a patient matched to a PI-CAI copy (R1b).
#   Agreement: a top-3 ProViCNet candidate centre within 10 mm of the finding position.
#   Test: logistic GEE clustered by patient, outcome = clinically significant finding, predictors = agreement and zone as recorded.
#   Replicated if the odds ratio for agreement is above 1 and its two-sided 95% CI excludes 1.
#   Also reported, without tests: proportions with Wilson intervals, unadjusted odds ratio, exclusions; a model adding log PSA density
#   (PI-CAI clinical table).
#   python analysis/R5_prostatex_agreement.py
import os, sys, warnings
import numpy as np, pandas as pd, SimpleITK as sitk
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import gee_fit

REP = 'outputs/replication/'
RAW = REP + 'raw/ProstateX-TrainingLesionInformationv2/'
ROOT = os.environ.get('PROSTATE_DATA_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
D = os.path.join(ROOT, 'PROSTATEx_replication')
F = pd.read_csv(RAW + 'ProstateX-Findings-Train.csv')
M = pd.read_csv(REP + 'match_table_R1b.csv')
Q = pd.read_csv(REP + 'R2_inputs_log.csv').set_index('ProxID')
C = pd.read_csv(REP + 'R4_candidates_ProViCNet.csv')
ms = pd.read_csv(os.environ.get('PROSTATE_DATA_ROOT', '../..') + '/PI-CAI_Validation/picai_labels-main/clinical_information/marksheet.csv')
RAD = float(os.environ.get('XRAD', '10'))
TOPK = int(os.environ.get('XTOPK', '3'))
TAG = os.environ.get('XTAG', '')

have = set(C.ProxID)
top = {p: g[['CoM_LPS_X_mm', 'CoM_LPS_Y_mm', 'CoM_LPS_Z_mm']].to_numpy(float) for p, g in C[(C.Candidate_Rank > 0) & (C.Candidate_Rank <= TOPK)].groupby('ProxID')}
n_all = len(F)
rows, ex = [], dict(patient_not_matched=0, input_failed=0, content_rule=0, no_probability_map=0, position_outside_image=0)
for _, f in F.iterrows():
    p = f.ProxID
    if p not in set(M.ProxID):
        ex['patient_not_matched'] += 1; continue
    if Q.loc[p, 'ok'] != 1:
        ex['input_failed'] += 1; continue
    if Q.loc[p, 'qc_fail'] == 1:
        ex['content_rule'] += 1; continue
    if p not in have:
        ex['no_probability_map'] += 1; continue
    pos = np.array([float(v) for v in f.pos.split()])
    rd = sitk.ImageFileReader(); rd.SetFileName(os.path.join(D, 'imagesTs', f'PX_{p}_0000.nii.gz')); rd.ReadImageInformation()
    ref = sitk.Image([int(v) for v in rd.GetSize()], sitk.sitkUInt8)
    ref.SetOrigin(rd.GetOrigin()); ref.SetSpacing(rd.GetSpacing()); ref.SetDirection(rd.GetDirection())
    idx = np.array(ref.TransformPhysicalPointToContinuousIndex(pos.tolist()))
    if not (np.all(idx >= -0.5) and np.all(idx <= np.array(rd.GetSize()) - 0.5)):
        ex['position_outside_image'] += 1; continue
    c = top.get(p)
    dmin = float(np.sqrt(((c - pos[None]) ** 2).sum(1)).min()) if c is not None else np.inf
    rows.append(dict(ProxID=p, fid=f.fid, zone=f.zone, y=int(bool(f.ClinSig)), dist_mm=dmin, agree=int(dmin <= RAD), n_candidates=0 if c is None else len(c)))
T = pd.DataFrame(rows).merge(M[['ProxID', 'patient_id', 'study_id']], on='ProxID').merge(ms[['patient_id', 'study_id', 'psa', 'psad', 'prostate_volume']], on=['patient_id', 'study_id'], how='left')
T['psad_use'] = T.psad.where(T.psad.notna(), T.psa / T.prostate_volume)
T['lpsad'] = np.log(T.psad_use.where(T.psad_use > 0))
T.to_csv(REP + f'R5_findings{TAG}.csv', index=False)
out = []


def wil(k, n):
    lo, hi = proportion_confint(k, n, method='wilson') if n else (np.nan, np.nan)
    return f'{k}/{n} = {100 * k / max(n, 1):.1f}% [{100 * lo:.1f}-{100 * hi:.1f}]'


def model(sub, zone, psad, lab):
    X = pd.DataFrame({'agree': sub.agree.to_numpy(float)})
    if zone:
        for z in [z for z in sorted(sub.zone.unique()) if z != 'PZ']:
            X['zone_' + z] = (sub.zone == z).to_numpy(float)
    if psad:
        X['lpsad'] = sub.lpsad.to_numpy(float)
    r = gee_fit(sub.y.to_numpy(), sm.add_constant(X), sub.ProxID.to_numpy())
    b, se = float(r.params['agree']), float(r.bse['agree'])
    o, lo, hi, pv = np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), float(r.pvalues['agree'])
    print(f'  {lab:58s} n={len(sub):3d} | OR {o:.2f} [{lo:.2f}-{hi:.2f}] p={pv:.4f} | working correlation: {r.working}')
    out.append(dict(analysis=lab, n=len(sub), patients=sub.ProxID.nunique(), cspca=int(sub.y.sum()), OR=o, lo=lo, hi=hi, p=pv, working=r.working))
    return o, lo, hi


print(f'PROSTATEx training findings {n_all} in {F.ProxID.nunique()} patients | excluded: {ex} | analysed {len(T)} findings in {T.ProxID.nunique()} patients')
print(f'agreement radius {RAD:g} mm, top-{TOPK} candidates | clinically significant {int(T.y.sum())} ({T.y.mean():.1%}) | agreement in {T.agree.mean():.1%} | zones {T.zone.value_counts().to_dict()}')
a1, a0 = T[T.agree == 1], T[T.agree == 0]
print(f'  clinically significant with agreement {wil(int(a1.y.sum()), len(a1))} | without {wil(int(a0.y.sum()), len(a0))}')
print(f'  agreement among clinically significant findings {wil(int(T[T.y == 1].agree.sum()), int((T.y == 1).sum()))} | among other findings {wil(int(T[T.y == 0].agree.sum()), int((T.y == 0).sum()))}')
o, lo, hi = model(T, True, False, 'main analysis: agreement + zone')
print('  ->', 'REPLICATED' if (o > 1 and lo > 1) else 'NOT REPLICATED', '(criterion: OR > 1 and 95% CI excluding 1)')
model(T, False, False, 'unadjusted (reported without a test)')
S = T[np.isfinite(T.lpsad)]
model(S, True, True, f'agreement + zone + log PSA density')
print(f'  PSA density available for {len(S)} of {len(T)} findings')
for z in ('PZ', 'TZ', 'AS'):
    q = T[T.zone == z]
    print(f'  zone {z}: n={len(q)} | with agreement {wil(int(q[q.agree == 1].y.sum()), int((q.agree == 1).sum()))} | without {wil(int(q[q.agree == 0].y.sum()), int((q.agree == 0).sum()))}')
res = pd.DataFrame(out)
res['agree_n'], res['agree_cspca'], res['disagree_n'], res['disagree_cspca'] = len(a1), int(a1.y.sum()), len(a0), int(a0.y.sum())
for k, v in ex.items():
    res['excl_' + k] = v
res.to_csv(REP + f'R5_results{TAG}.csv', index=False)
