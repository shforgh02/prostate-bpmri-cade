# Assembles one row per model from the result files of A1, A2, A3 and B6 (same --tag) into a summary table.
# Run from the repository root:
#   python analysis/C1_summary_table.py --tag final
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='final')
a = ap.parse_args()
O = 'outputs/'
A1, A2, A3, B6 = (pd.read_csv(O + f'{n}_results_{a.tag}.csv') for n in ('A1', 'A2', 'A3', 'B6'))
rows = []
for m in sorted(A2.model.unique()):
    r = {'model': m}
    x = A1[(A1.model == m) & (A1.tol_mm == 10)].set_index('subset')
    r['top3_all'] = f"{x.loc['all csPCa', 'observed']}/{x.loc['all csPCa', 'n']} vs {x.loc['all csPCa', 'perm_expected']} by chance"
    r['rad_miss'] = f"{x.loc['rad target MISS', 'observed']}/{x.loc['rad target MISS', 'n']} vs {x.loc['rad target MISS', 'perm_expected']} (p={x.loc['rad target MISS', 'perm_p']:.3f})"
    y = A2[A2.model == m].set_index('analysis')
    p = y.loc['OFF-TARGET, 10 mm (main)']
    r['offtarget_rate'] = f"{p.value:.1%} ({int(p.cspca)}/{int(p.n)}) vs null {p.null_mean:.1%} [{p.lo:.1%}-{p.hi:.1%}]"
    r['offtarget_p'] = round(p.p, 4)
    r['offtarget_ratio_to_null'] = round(p.value / p.null_mean, 2)
    w = y.loc['within-patient off-target AUROC']
    r['within_patient_AUROC'] = f"{w.value:.3f} [{w.lo:.3f}-{w.hi:.3f}], null {w.null_mean:.3f}, p={w.p:.4f}"
    c = y.loc['near radiologist target, 10 mm (positive control)']
    r['near_target_rate'] = f"{c.value:.1%} vs null {c.null_mean:.1%} (p={c.p:.4f})"
    b = B6[B6.model == m].set_index('subset')
    for k, lab in (('all sampled targets, score 3-5', 'agreement_OR_all'), ('score 3', 'agreement_OR_score3')):
        q = b.loc[k]
        r[lab] = f"{q.OR:.2f} [{q.lo:.2f}-{q.hi:.2f}]; {q.agree_cspca}/{q.agree_n} vs {q.disagree_cspca}/{q.disagree_n}"
    z = A3[A3.model == m].set_index('analysis')
    r['exam_AUROC'] = f"{z.loc['AUROC alone, all exams', 'value']:.3f}"
    g = z.loc['gain over score + PSA density']
    r['gain_over_score_PSAD'] = f"{g.value:+.3f} [{g.lo:+.3f}, {g.hi:+.3f}]"
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(O + f'SUMMARY_TABLE_{a.tag}.csv', index=False)
pd.set_option('display.width', 250, 'display.max_colwidth', 70)
print(T.set_index('model').T.to_string())
