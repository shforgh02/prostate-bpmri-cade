# X5: robustness of the per-target agreement finding (B6), all three models.
#  - unadjusted vs adjusted OR; subgroups by PSA density, PSA, target volume, number of targets, age, field strength
#  - alternative outcomes: any cancer (csPCa or grade group 1) vs benign; csPCa vs grade group 1 among cancer-positive targets
#   python analysis/extended/X5_agreement_robustness.py > outputs/extended/X5.txt
import sys
sys.path.insert(0, 'analysis/extended')
from xcommon import *

T = exam_table().set_index('Session_ID')
tg = pd.read_csv(O + 'B6_targets_final_qc.csv')
tg = tg[(tg.path != 'unsampled') & tg.score.isin([3, 4, 5]) & np.isfinite(tg.lpsad)].copy()
tg['y'] = (tg.path == 'csPCa').astype(int)
tg['s4'], tg['s5'] = (tg.score == 4).astype(int), (tg.score == 5).astype(int)
tg['age'] = T.loc[tg.Session_ID, 'age'].to_numpy()
tg['field'] = T.loc[tg.Session_ID, 'field'].to_numpy()
tg['psad'] = np.exp(tg.lpsad)
v = tg.vol_cc.quantile([1 / 3, 2 / 3]).to_numpy()
print(f'{len(tg)} sampled targets with score 3-5 and PSA density (as in B6); OR for csPCa at the target when a top-3 candidate lies within 10 mm')
rows = []


def run(lab, d, outcome='y', adjust=True):
    cells = []
    for m in MODELS:
        x = [f'agree_{m}'] + ((['s4', 's5'] if d.score.nunique() > 1 else []) + ['lpsad'] if adjust else [])
        try:
            o, lo, hi, p = gee_or(d, outcome, x, 'Patient_ID')
        except Exception:
            o = lo = hi = p = np.nan
        rows.append(dict(analysis=lab, n=len(d), model=NAME[m], OR=o, lo=lo, hi=hi, p=p))
        cells.append(f'{NAME[m]} {o:.2f} [{lo:.2f}-{hi:.2f}]')
    print(f'  {lab:46s} n={len(d):3d} | ' + ' | '.join(cells))


run('all, unadjusted', tg, adjust=False)
run('all, adjusted (main analysis)', tg)
for lab, msk in (('PSA density < 0.15', tg.psad < 0.15), ('PSA density >= 0.15', tg.psad >= 0.15), ('PSA < 10', tg.PSA < 10), ('PSA >= 10', tg.PSA >= 10),
                 (f'target volume < {v[0]:.2f} mL', tg.vol_cc < v[0]), (f'target volume {v[0]:.2f}-{v[1]:.2f} mL', tg.vol_cc.between(v[0], v[1], inclusive='left')),
                 (f'target volume >= {v[1]:.2f} mL', tg.vol_cc >= v[1]), ('single-target examinations', tg.n_targets == 1), ('2 or more targets', tg.n_targets >= 2),
                 ('age < 65', tg.age < 65), ('age >= 65', tg.age >= 65), ('3 T only', tg.field == 3)):
    run(lab, tg[msk])
tg['anyca'] = tg.path.isin(['csPCa', 'GG1']).astype(int)
print('\nAlternative outcomes')
run('any cancer (csPCa or GG1) vs benign', tg, outcome='anyca')
run('csPCa vs GG1, cancer-positive targets only', tg[tg.anyca == 1])
pd.DataFrame(rows).to_csv(R + 'X5_agreement_robustness.csv', index=False)
print('\nNotes: GEE, binomial, exchangeable, clustered by patient; adjusted for suspicion score (when the subgroup has more than one score) '
      'and log PSA density. Subgroups without multiplicity adjustment.')
