# Tables 2 and 3 of the manuscript. Numbers are read from outputs/ files only.
#   python analysis/D9_tables.py   -> tables/Table2_models.csv, Table3_discrimination.csv, TABLES_MAIN.md
import numpy as np, pandas as pd

O, TAB = 'outputs/', 'tables/'
KEYS = [('3D_SegResNet', 'SegResNet'), ('nnUNet_official', 'nnU-Net'), ('ProViCNet', 'ProViCNet')]
TRAIN = {'3D_SegResNet': 'PI-CAI training set, three of five folds (900 examinations); trained for this study',
         'nnUNet_official': 'PI-CAI training set (1,500 examinations); public five-network ensemble',
         'ProViCNet': '1,404 patients from another institution; foundation model backbone; published weights'}


def P(p):
    if p is None or not np.isfinite(p):
        return ''
    if round(p, 3) < 0.001:
        return '<.001'
    return (f'{p:.3f}' if p < 0.01 else f'{p:.2f}').lstrip('0')


def ci(v, lo, hi, d=2):
    return f'{v:.{d}f} ({lo:.{d}f}, {hi:.{d}f})'


a1 = pd.read_csv(O + 'A1_results_final_qc.csv'); a1 = a1[(a1.tol_mm == 10.0) & (a1.subset == 'all csPCa')].set_index('model')
b6 = pd.read_csv(O + 'B6_results_final_qc.csv').set_index(['model', 'subset'])
a2 = pd.read_csv(O + 'A2_results_final_qc.csv'); a2 = a2[a2.analysis == 'OFF-TARGET, 10 mm (main)'].set_index('model')
h = pd.read_csv(O + 'C4_holm.csv'); h = h[h.tag == 'final_qc'].set_index('model')
a3 = pd.read_csv(O + 'A3_results_final_qc.csv')
# Holm adjustment of the agreement ORs (all sampled targets, score 3-5) across the three models
from statsmodels.stats.multitest import multipletests
_pa = [float(b6.loc[(k, 'all sampled targets, score 3-5'), 'p']) for k, _ in KEYS]
AGREE_HOLM = dict(zip([k for k, _ in KEYS], multipletests(_pa, method='holm')[1]))
assert [P(AGREE_HOLM[k]) for k, _ in KEYS] == ['<.001', '.008', '.003'], AGREE_HOLM
rows = []
for k, n in KEYS:
    r1, r2 = a1.loc[k], a2.loc[k]
    ba, b3 = b6.loc[(k, 'all sampled targets, score 3-5')], b6.loc[(k, 'score 3')]
    alone = a3[(a3.model == k) & (a3.analysis == 'AUROC alone, all exams')].iloc[0]
    gain = a3[(a3.model == k) & (a3.analysis == 'gain over score + PSA density')].iloc[0]
    rows.append({'Model': n, 'Training data': TRAIN[k],
                 'Localization of 387 cancers, top 3, 10 mm (chance)': f'{int(r1.observed)} ({100 * r1.observed / r1.n:.1f}%) ({100 * r1.perm_expected / r1.n:.1f}%)',
                 f'Agreement OR, all targets (n = {int(ba.n)})': ci(ba.OR, ba.lo, ba.hi), 'Agreement OR, all targets: P, Holm-adjusted': P(AGREE_HOLM[k]),
                 f'Agreement OR, score 3 targets (n = {int(b3.n)})': ci(b3.OR, b3.lo, b3.hi),
                 'csPCa in cores away from targets near a candidate (expected)': f'{100 * r2.value:.1f}% ({int(r2.cspca)} of {int(r2.n)}) ({100 * r2.null_mean:.1f}%)',
                 'P, one-sided': P(h.loc[k, 'p_raw']), 'P, Holm-adjusted': P(h.loc[k, 'p_holm']),
                 'Examination AUC alone (n = 757)': f'{alone.value:.3f} ({alone.lo:.3f}, {alone.hi:.3f})',
                 'AUC gain over score + PSA density (n = 719)': f'{gain.value:+.3f} ({gain.lo:+.3f}, {gain.hi:+.3f})'.replace('-0.000', '0.000').replace('+0.000', '0.000')})
T2 = pd.DataFrame(rows)
T2.to_csv(TAB + 'Table2_models.csv', index=False)

c5 = pd.read_csv(O + 'C5_results_final_qc.csv')
ref = a3[a3.model == 'reference'].set_index('analysis')
NA, N3 = b6.loc[('3D_SegResNet', 'all sampled targets, score 3-5')], b6.loc[('3D_SegResNet', 'score 3')]
HA = f'Target level, all (n = {int(NA.n)}; {int(NA.agree_cspca + NA.disagree_cspca)} csPCa): AUC'
H3 = f'Target level, score 3 (n = {int(N3.n)}; {int(N3.agree_cspca + N3.disagree_cspca)} csPCa): AUC'
def zz(t):
    return t.replace('-0.000', '0.000').replace('+0.000', '0.000')


rows = []
for lab, c5lab, k in [('Suspicion score + PSA density (reference)', 'base: suspicion score + log PSA density', None)] + \
                     [(f'+ {n}', f'+ {n} agrees', kk) for kk, n in KEYS] + [('+ number of agreeing models (0–3)', '+ number of agreeing models', 'n')]:
    t_all = c5[(c5.scope == 'all targets (score 3-5)') & (c5.model == c5lab)].iloc[0]
    t_s3 = c5[(c5.scope == 'score 3 targets') & (c5.model == c5lab)].iloc[0]
    row = {'Predictors': lab,
           HA: f'{t_all.auc:.3f}',
           'Gain (95% CI)': '' if k is None else zz(f'{t_all.gain:+.3f} ({t_all.lo:+.3f}, {t_all.hi:+.3f})'),
           H3: f'{t_s3.auc:.3f}',
           'Gain (95% CI) ': '' if k is None else zz(f'{t_s3.gain:+.3f} ({t_s3.lo:+.3f}, {t_s3.hi:+.3f})')}
    if k is None:
        row['Examination level (n = 719; 396 csPCa): AUC'] = f"{ref.loc['score + PSA density', 'value']:.3f}"
        row['Gain (95% CI)  '] = ''
    elif k == 'n':
        row['Examination level (n = 719; 396 csPCa): AUC'] = '— (a)'
        row['Gain (95% CI)  '] = ''
    else:
        g = a3[(a3.model == k) & (a3.analysis == 'gain over score + PSA density')].iloc[0]
        row['Examination level (n = 719; 396 csPCa): AUC'] = f"{ref.loc['score + PSA density', 'value'] + g.value:.3f}"
        row['Gain (95% CI)  '] = f'{g.value:+.3f} ({g.lo:+.3f}, {g.hi:+.3f})'.replace('-0.000', '0.000')
    rows.append(row)
T3 = pd.DataFrame(rows)
T3.to_csv(TAB + 'Table3_discrimination.csv', index=False)
T1 = pd.read_csv(TAB + 'Table1_characteristics.csv').fillna('')
_ep = pd.read_csv('results/tables/EXAMINATION_LEVEL_ENDPOINTS.csv')
_ep = _ep[~_ep.Session_ID.isin(pd.read_csv(O + 'EXCLUDED_SESSIONS_v2.csv').Session_ID)]
N_NOPOS = int(((_ep.Is_csPCa == 1) & (_ep.Has_Valid_3D_Trajectory != 1)).sum())          # men with csPCa but no csPCa-positive core with a recorded position
assert N_NOPOS == 24, N_NOPOS
# Table 3 note values, generated from the result files
_r = ref['value']
_lr = {k: a3[(a3.model == k) & (a3.analysis == 'gain over score + PSA density')].p.iloc[0] for k, _ in KEYS}
_x4 = pd.read_csv('outputs/extended/X4a_exam_level.csv').set_index('model').loc['score + PSA density + number of models agreeing']
NOTE3 = ('Target level: sampled radiologist targets with a suspicion score of 3–5 and PSA density; reference model = suspicion score of the target plus log PSA density; '
         'logistic regression cross-fitted by patient (five folds); gains with 95% CIs from 2,000 bootstrap resamples of patients; score 3 values evaluate the same predictions '
         'within score 3 targets. Examination level: examinations with a highest suspicion score of 3–5 and PSA density; model term = highest candidate probability; '
         f"single predictors: suspicion score, AUC {_r['suspicion score']:.3f}; PSA density, AUC {_r['PSA density']:.3f}; likelihood ratio test for adding the model term: "
         f"P = {P(_lr['3D_SegResNet'])} (SegResNet), {P(_lr['nnUNet_official'])} (nnU-Net), and {P(_lr['ProViCNet'])} (ProViCNet). (a) The number of agreeing models is defined "
         'per target; an examination-level version (number of models that agree with any target) is given in Table S13A '
         f'(717 men; AUC gain, {_x4.gain:+.3f}; 95% CI: {_x4.lo:+.3f}, {_x4.hi:+.3f}).').replace('-0.', '−0.')
md = ('# Tables 1-3 (generated by analysis/D3_table1.py and D9_tables.py)\n\n'
      '## Table 1: Characteristics of the 757 Men\n\n' + T1.to_markdown(index=False) +
      f'\n\nAge is taken from the DICOM header of the analyzed T2-weighted series. In {N_NOPOS} men with csPCa, every csPCa-positive core had placeholder coordinates. '
      'IQR = interquartile range, PSA = prostate-specific antigen.\n\n'
      '## Table 2: Results for Each Model\n\n' + T2.to_markdown(index=False) +
      '\n\nAgreement OR = odds ratio for csPCa at a radiologist target when the center of one of the three highest-ranked candidates lay within 10 mm of the target center, '
      'adjusted for the suspicion score of the target and log PSA density (generalized estimating equations clustered by patient). Localization: expected percentage when the '
      'candidates of each man are replaced by those of every other man with the same number of candidates (Poisson binomial distribution). Cores away from targets: systematic '
      'cores more than 10 mm from every targeted core and from the target center; expected rate from 2,000 random exchanges of candidate sets; one-sided P values, also adjusted '
      'across the three models with the Holm method. AUC gain in 719 men from cross-fitted logistic models with 95% CIs from 2,000 bootstrap resamples of patients. '
      'csPCa = clinically significant prostate cancer, PSA = prostate-specific antigen.\n\n'
      '## Table 3: Discrimination of csPCa at the Target Level and the Examination Level\n\n' + T3.to_markdown(index=False) +
      '\n\n' + NOTE3 + '\n')
assert '\n\n' not in NOTE3
open(TAB + 'TABLES_MAIN.md', 'w', encoding='utf-8').write(md)
print(T2.T.to_string()); print(); print(T3.to_string(index=False))
