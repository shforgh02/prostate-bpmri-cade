# Equivocal (suspicion score 3) examinations: proportion with any csPCa (targeted or systematic cores) by PSA density
# and by how many models agree with a radiologist target (top-3 candidate centre within 10 mm of the target centre).
# Needs outputs/B6_targets_<tag>.csv.   python analysis/C3_score3_stratification.py --tag final_qc
import argparse, numpy as np, pandas as pd, warnings
from statsmodels.stats.proportion import proportion_confint as ci
warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='final_qc'); a = ap.parse_args()
O = 'outputs/'
T = pd.read_csv(O + f'B6_targets_{a.tag}.csv'); ep = pd.read_csv('results/tables/EXAMINATION_LEVEL_ENDPOINTS.csv')
M = [c[6:] for c in T.columns if c.startswith('agree_')]
g = T.groupby('Session_ID').agg(lpsad=('lpsad', 'first'), **{m: ('agree_' + m, 'max') for m in M}).reset_index()
E = ep.merge(g, on='Session_ID'); E['n_agree'] = E[M].sum(1); E['psad'] = np.exp(E.lpsad)
S3 = E[(E.Exact_Exam_UCLA_Score == 3) & np.isfinite(E.lpsad)]
rows = []
def row(lab, d):
    k, n = int(d.Is_csPCa.sum()), len(d)
    lo, hi = ci(k, n, method='wilson') if n else (np.nan, np.nan)
    print(f'  {lab:46s} n={n:3d} ({n / len(S3):4.0%}) | any csPCa {k:3d} = {(k / n if n else np.nan):5.1%} [{lo:.1%}-{hi:.1%}] | share of score-3 cancers {k / max(S3.Is_csPCa.sum(), 1):4.0%}')
    rows.append(dict(group=lab, n=n, share_of_exams=n / len(S3), cspca=k, rate=k / n if n else np.nan, lo=lo, hi=hi, share_of_cancers=k / max(S3.Is_csPCa.sum(), 1)))
print(f'score 3 exams with a contoured target and PSA density: n={len(S3)}, any csPCa {int(S3.Is_csPCa.sum())} ({S3.Is_csPCa.mean():.1%}) | models {M}')
row('all', S3)
for lab, d in (('PSAD < 0.10', S3[S3.psad < 0.10]), ('PSAD < 0.15', S3[S3.psad < 0.15]), ('PSAD >= 0.15', S3[S3.psad >= 0.15])): row(lab, d)
for k in range(len(M) + 1): row(f'{k} of {len(M)} models agree', S3[S3.n_agree == k])
for m in M:
    row(f'{m} agrees', S3[S3[m] == 1]); row(f'{m} does not agree', S3[S3[m] == 0])
    row(f'{m} does not agree & PSAD < 0.15', S3[(S3[m] == 0) & (S3.psad < 0.15)]); row(f'{m} agrees & PSAD >= 0.15', S3[(S3[m] == 1) & (S3.psad >= 0.15)])
row('no model agrees & PSAD < 0.15', S3[(S3.n_agree == 0) & (S3.psad < 0.15)]); row('all models agree & PSAD >= 0.15', S3[(S3.n_agree == len(M)) & (S3.psad >= 0.15)])
pd.DataFrame(rows).to_csv(O + f'C3_results_{a.tag}.csv', index=False)
