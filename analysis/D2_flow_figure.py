# Figure 1: study design (a models, b study sample, c analyses, d external replication).
# Cohort numbers are parsed from outputs/FLOW_NUMBERS.txt and outputs/B6_final_qc.txt; model facts are fixed text
# (PI-CAI fold sizes from the SegResNet training manifest; nnU-Net and ProViCNet from their publications).
#   python analysis/D2_flow_figure.py      -> figures/Figure1_flow.png (300 dpi) and .pdf
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

O, FIG = 'outputs/', 'figures/'
T = open(O + 'FLOW_NUMBERS.txt', encoding='utf-8').read()
B6 = open(O + 'B6_final_qc.txt', encoding='utf-8').read()


def grab(pat, src=T):
    m = re.search(pat, src, flags=re.M)
    assert m, 'pattern not found: ' + pat
    return [g.replace(',', '') for g in m.groups()]


n800, = grab(r'^(\d+) paired examinations')
x1, x1_b, x1_none = grab(r'-> (\d+) excluded: selected high-b image with b < 1000 s/mm2 \((\d+)\) or b-value missing in header \((\d+)\)')
x2, = grab(r'-> (\d+) excluded: unusable diffusion inside the prostate')
n757, cs, ev = grab(r'^(\d+) examinations \(main cohort, tag final_qc\): (\d+) csPCa; (\d+) evaluable csPCa')
cores, targ, cpos, sysc, sysex, off, offpos = grab(r'cores in the 757 cohort: ([\d,]+) \(([\d,]+) targeted; ([\d,]+) csPCa-positive\); systematic cores with valid trajectory ([\d,]+) in (\d+) exams, of which off-target ([\d,]+) \((\d+) csPCa-positive\)')
ntar, nsamp = grab(r'targets (\d+) \| sampled (\d+)', B6)
n843, = grab(r'all sampled targets, score 3-5\s+n=\s*(\d+)', B6)
assert int(n800) - int(x1) - int(x2) == int(n757)


def f(n):
    return f'{int(n):,}'


INK, INK2, EDGE = '#1f1f1e', '#4a4a47', '#5f5e5a'
BLUE, ORANGE, GREEN, NAVY = '#2a78d6', '#eb6834', '#1baf7a', '#1f3a6e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(7.2, 9.4))
ax.set_xlim(0, 1)
ax.set_ylim(-0.064, 1)
ax.axis('off')


def box(xc, yc, w, h, title, body='', fill='#ffffff', ec=EDGE, tc=INK, ts=8.2, bs=7.0, lw=1.0):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle='round,pad=0.004,rounding_size=0.012', fc=fill, ec=ec, lw=lw))
    if body:
        ax.text(xc, yc + h / 2 - 0.014, title, ha='center', va='top', fontsize=ts, color=tc, fontweight='bold')
        ax.text(xc, yc + h / 2 - 0.040 - 0.022 * title.count('\n'), body, ha='center', va='top', fontsize=bs, color=INK2, linespacing=1.35)
    else:
        ax.text(xc, yc, title, ha='center', va='center', fontsize=ts, color=tc, fontweight='bold')


def arrow(x0, y0, x1_, y1_, **kw):
    ax.annotate('', xy=(x1_, y1_), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=0.9, mutation_scale=9, **kw))


def panel(letter, y):
    ax.text(0.0, y, letter, fontsize=12, fontweight='bold', color=INK, va='center', ha='left')


# ---------------------------------------------------------------- a: models
panel('a', 0.985)
ax.text(0.045, 0.985, 'Models', fontsize=9, fontweight='bold', color=INK, va='center')
yA, hA, wA = 0.905, 0.108, 0.30
for xc, col, title, body in (
        (0.185, BLUE, 'SegResNet', 'Trained for this study on PI-CAI\ntraining: 900 examinations\nvalidation: 304; internal test: 296'),
        (0.515, ORANGE, 'nnU-Net', 'Public PI-CAI baseline\nfive-fold ensemble\ntrained on 1,500 examinations'),
        (0.845, GREEN, 'ProViCNet', 'Published weights\nfoundation-model backbone\ntrained on 1,404 patients\nat another institution')):
    box(xc, yA, wA, hA, title, body, ec=col, tc=col, lw=1.4)
    arrow(xc, yA - hA / 2, xc, 0.795)
yT = 0.770
box(0.515, yT, 0.96, 0.046, 'Applied without retraining to the same T2-weighted, ADC, and high-b-value images', fill='#f3f2ee', ts=7.8)

# ---------------------------------------------------------------- b: study sample
panel('b', 0.715)
ax.text(0.045, 0.715, 'Study sample', fontsize=9, fontweight='bold', color=INK, va='center')
MX, MW, EX, EW, H = 0.27, 0.44, 0.765, 0.45, 0.070
Y = dict(a=0.655, b=0.520, c=0.415, d=0.310)
arrow(MX, yT - 0.023, MX, Y['a'] + H / 2)
box(MX, Y['a'], MW, H, f'{f(n800)} examinations', 'UCLA Prostate-MRI-US-Biopsy, one per man', ec=NAVY)
box(MX, Y['b'], MW, H, f'{f(n757)} examinations analyzed', 'MRI–US fusion biopsy with tracked cores', ec=NAVY)
box(MX, Y['c'], MW, H, f'{f(cs)} with csPCa', 'grade group ≥ 2 in any core', ec=NAVY)
box(MX, Y['d'], MW, H, f'{f(ev)} evaluable cancers', 'csPCa core with a valid needle trajectory', ec=NAVY)
for a_, b_ in (('a', 'b'), ('b', 'c'), ('c', 'd')):
    arrow(MX, Y[a_] - H / 2, MX, Y[b_] + H / 2)
ym = (Y['a'] + Y['b']) / 2
box(EX, ym + 0.012, EW, 0.098, f'{int(x1) + int(x2)} excluded',
    f'high-b-value image with b < 1000 sec/mm² or\nno b value in the header: {x1}\ndiffusion images unusable inside the prostate: {x2}', fill='#fdf3f0', ec='#c0392b')
arrow(MX, ym, EX - EW / 2, ym)
ys = (Y['c'] + Y['d']) / 2
box(EX, ys, EW, 0.118, 'Reference standard',
    f'{f(cores)} tracked cores ({f(targ)} targeted)\n{f(ntar)} radiologist targets ({f(nsamp)} sampled)\n'
    f'{f(off)} systematic cores away from targets,\n{f(offpos)} with csPCa', fill='#f3f2ee')
ax.plot([MX + MW / 2, EX - EW / 2], [Y['b'], ys + 0.03], color=EDGE, lw=0.7, ls=(0, (2, 2)))

# ---------------------------------------------------------------- c: analyses
panel('c', 0.252)
ax.text(0.045, 0.252, 'Analyses', fontsize=9, fontweight='bold', color=INK, va='center')
yC, hC, wC = 0.140, 0.128, 0.225
for xc, title, body in (
        (0.1425, 'Localization\nof cancer', f'candidate within 10 mm\nof a csPCa core, compared\nwith chance ({f(ev)} cancers)'),
        (0.3875, 'Agreement\nat targets', f'csPCa at targets with and\nwithout a candidate within\n10 mm ({f(n843)} targets)'),
        (0.6325, 'Cores away\nfrom targets', f'csPCa in cores near a\ncandidate, compared\nwith chance'),
        (0.8775, 'Examination\nlevel', 'discrimination added to\nsuspicion score and\nPSA density')):
    box(xc, yC, wC, hC, title, body, bs=6.8, ts=7.8)
yb = 0.232
ax.plot([MX, MX], [Y['d'] - H / 2, yb], color=EDGE, lw=0.9)
ax.plot([0.1425, 0.8775], [yb, yb], color=EDGE, lw=0.9)
for xc in (0.1425, 0.3875, 0.6325, 0.8775):
    arrow(xc, yb, xc, yC + hC / 2)

# ---------------------------------------------------------------- d: external replication
import pandas as pd
RP = 'outputs/replication/'
_all = pd.read_csv(RP + 'raw/ProstateX-TrainingLesionInformationv2/ProstateX-Findings-Train.csv')
_m, _t = pd.read_csv(RP + 'match_table_R1b.csv'), pd.read_csv(RP + 'R5_findings.csv')
panel('d', 0.044)
ax.text(0.045, 0.044, 'External replication', fontsize=9, fontweight='bold', color=INK, va='center')
box(0.515, -0.018, 0.96, 0.078, f'PROSTATEx training set: {_t.ProxID.nunique()} of {_all.ProxID.nunique()} men, {len(_t)} MRI findings sampled by MR-guided biopsy',
    'ProViCNet only (not trained on these examinations): clinically significant cancer at findings' + chr(10) +
    'with and without a candidate within 10 mm', fill='#f3f2ee', ts=7.2, bs=6.6)
fig.savefig(FIG + 'Figure1_flow.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(FIG + 'Figure1_flow.pdf', bbox_inches='tight', facecolor='white')
print('saved', n800, x1, x2, n757, cs, ev, cores, targ, ntar, nsamp, n843, off, offpos)
