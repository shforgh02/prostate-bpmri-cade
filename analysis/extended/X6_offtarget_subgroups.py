# X6: the analysis of cores away from the targets (analysis/A2, unchanged) repeated within subgroups, by passing exclusion lists
# (EXCLUDED_SESSIONS_v2 + every examination outside the subgroup). Subgroups: score 3, score 4-5, PSA density < 0.15 and >= 0.15,
# 3 T only, Siemens only. A2 writes A2_results_xsub_<name>.csv into outputs/; this script moves them into outputs/extended/.
# The agreement-radius runs (X7) use X7_b6_agreement_radius.py with XRAD=5 and XRAD=15 (tags x_rad5, x_rad15).
#   python analysis/extended/X6_offtarget_subgroups.py
import os, shutil, subprocess, sys
import numpy as np, pandas as pd

R = 'outputs/extended/'
O = 'outputs/'
T = pd.read_csv(R + 'X0_exam_table.csv')
v2 = pd.read_csv(O + 'EXCLUDED_SESSIONS_v2.csv')
SUB = {'score3': T.score == 3, 'score45': T.score.isin([4, 5]), 'psadlow': T.psad < 0.15, 'psadhigh': T.psad >= 0.15,
       'field3T': T.field == 3, 'siemens': T.vendor.str.upper().str.contains('SIEMENS')}
for k, m in SUB.items():
    ex = pd.concat([v2[['Session_ID']], T.loc[~m, ['Session_ID']]]).drop_duplicates()
    ex.to_csv(R + f'X6_exclude_{k}.csv', index=False)
    with open(R + f'X6_{k}.txt', 'w', encoding='utf-8') as f:
        subprocess.run([sys.executable, 'analysis/A2_core_level_off_target.py', '--candidates', O + 'CANDIDATES_3MODELS.csv',
                        '--exclude', R + f'X6_exclude_{k}.csv', '--tag', f'xsub_{k}'], stdout=f, stderr=subprocess.STDOUT, check=True)
    shutil.move(O + f'A2_results_xsub_{k}.csv', R + f'A2_results_xsub_{k}.csv')
    print(k, 'done', flush=True)
