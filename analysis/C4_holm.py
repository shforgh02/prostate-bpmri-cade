# Holm correction of the three PRIMARY off-target p-values, per cohort.
#   python analysis/C4_holm.py      -> outputs/C4_holm.csv
import pandas as pd
from statsmodels.stats.multitest import multipletests

O = 'outputs/'
rows = []
for tag, cohort in (('final_qc', 'main cohort'), ('final', 'cohort before the content exclusion')):
    d = pd.read_csv(O + f'A2_results_{tag}.csv')
    d = d[d.analysis == 'OFF-TARGET, 10 mm (main)'].reset_index(drop=True)
    assert len(d) == 3, (tag, len(d))
    rej, ph, _, _ = multipletests(d.p, method='holm')
    for i, r in d.iterrows():
        rows.append(dict(cohort=cohort, tag=tag, model=r.model, n=r.n, cspca=r.cspca, value=r.value, null_mean=r.null_mean, p_raw=r.p, p_holm=ph[i], reject_005=bool(rej[i])))
out = pd.DataFrame(rows)
out.to_csv(O + 'C4_holm.csv', index=False)
print(out.to_string(index=False))
