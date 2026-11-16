# Combines the two technical exclusion rules into one list.
#   header rule  (B0): outputs/EXCLUDED_SESSIONS.csv
#   content rule (B7): rows with qc_fail = 1 in outputs/B7_input_content_qc.csv
# Run from the repository root:  python analysis/B7b_combine_exclusions.py   -> outputs/EXCLUDED_SESSIONS_v2.csv
import numpy as np, pandas as pd
h = pd.read_csv('outputs/EXCLUDED_SESSIONS.csv')
q = pd.read_csv('outputs/B7_input_content_qc.csv')
q = q[(q.qc_fail == 1) & ~q.Session_ID.isin(h.Session_ID)].copy()
q['reason'] = np.select([q.flag_saturated_hbv == 1, q.flag_coverage == 1, q.flag_saturated_adc == 1],
                        ['high-b image constant or saturated in the gland', 'diffusion does not cover the gland', 'ADC constant in the gland'], 'content check')
ep = pd.read_csv('results/tables/EXAMINATION_LEVEL_ENDPOINTS.csv')
ev = ep[(ep.Is_csPCa == 1) & (ep.Has_Valid_3D_Trajectory == 1)].Session_ID
q['evaluable_csPCa'] = q.Session_ID.isin(ev).astype(int)
x = pd.concat([h, q[['Session_ID', 'reason', 'evaluable_csPCa']]], ignore_index=True)
x.to_csv('outputs/EXCLUDED_SESSIONS_v2.csv', index=False)
print('header rule', len(h), '| content rule', len(q), '| total', len(x))
