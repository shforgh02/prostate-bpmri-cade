# Lists analysed examinations whose selected "high-b" image has b < 1000 or unknown b-value.
# Run from the repository root:  python analysis/B0_exclusions.py
import numpy as np, pandas as pd
hb = pd.read_csv('outputs/highb_series_audit.csv', low_memory=False)
ep = pd.read_csv('results/tables/EXAMINATION_LEVEL_ENDPOINTS.csv')
h = hb[hb.session.isin(ep.Session_ID)].copy(); assert len(h) == len(ep)
h['b'] = pd.to_numeric(h.bvals, errors='coerce')
x = h[(h.b < 1000) | h.b.isna()][['session', 'dwi_desc', 'b']].rename(columns={'session': 'Session_ID', 'b': 'b_value'})
x['reason'] = np.where(x.b_value.isna(), 'b-value missing in header', 'b < 1000')
ev = ep[(ep.Is_csPCa == 1) & (ep.Has_Valid_3D_Trajectory == 1)].Session_ID
x['evaluable_csPCa'] = x.Session_ID.isin(ev).astype(int)
x.to_csv('outputs/EXCLUDED_SESSIONS.csv', index=False)
print('excluded', len(x), '| with evaluable csPCa:', int(x.evaluable_csPCa.sum()), '| remaining', len(ep) - len(x))
