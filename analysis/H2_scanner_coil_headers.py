# Usage: python analysis/H2_scanner_coil_headers.py <MRI_US_Biopsy dir> <out csv>   (reads one DICOM header per series; about 4 minutes)
# Read-only. Needs pandas, numpy, scipy, scikit-learn, SimpleITK .
"""Scanner and coil header fields per UCLA session (one DICOM file of the T2 series and one of the ADC series). Read-only.
Usage: python -I ucla_scanner_coil.py <MRI_US_Biopsy dir> <out_csv>"""
import sys, os, re, glob, zipfile, collections, time
import xml.etree.ElementTree as ET
import pandas as pd
import SimpleITK as sitk
ROOT, OUT = sys.argv[1], sys.argv[2]
MAN = os.path.join(ROOT, "manifest-1694710246744")
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
def read_xlsx(path):
    z = zipfile.ZipFile(path); ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            ss.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]; rows = []
    for r in ET.fromstring(z.read(sheet)).iter("{%s}row" % NS["m"]):
        d = {}
        for c in r.findall("m:c", NS):
            col = re.match(r"[A-Z]+", c.get("r")).group(0); v = c.find("m:v", NS); t = c.get("t")
            d[col] = None if v is None else (ss[int(v.text)] if t == "s" else v.text)
        rows.append(d)
    hdr = rows[0]; cols = sorted(hdr, key=lambda k: (len(k), k))
    return pd.DataFrame([{hdr[c]: r.get(c) for c in cols} for r in rows[1:]])
meta = pd.read_csv(os.path.join(MAN, "metadata.csv")); meta.columns = [c.strip() for c in meta.columns]
bx = read_xlsx(os.path.join(ROOT, "TCIA-Biopsy-Data_2020-07-14.xlsx"))
sess = collections.OrderedDict()
for _, r in bx.iterrows():
    pid = str(r["Patient Number"]).strip(); uid = str(r.get("Series Instance UID (MRI)", "")).strip()
    if uid and uid not in ("nan", "None"): sess.setdefault(f"{pid}_{uid[-5:]}", uid)
rd = sitk.ImageFileReader(); rd.LoadPrivateTagsOn()
TAGS = dict(model="0008|1090", field="0018|0087", coil="0018|1250", coil_siemens="0051|100f", software="0018|1020", manufacturer="0008|0070", tx_coil="0018|1251")
def head(folder):
    f = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    if not f: return {}
    rd.SetFileName(f[len(f) // 2]); rd.ReadImageInformation()
    return {k: (rd.GetMetaData(t).strip() if rd.HasMetaDataKey(t) else "") for k, t in TAGS.items()}
rows = []; t0 = time.time()
for sid, uid in sess.items():
    t2 = meta[meta["Series UID"] == uid]
    if t2.empty: continue
    st = meta[meta["Study UID"] == t2.iloc[0]["Study UID"]]
    o = dict(Session_ID=sid)
    try:
        o.update({f"t2_{k}": v for k, v in head(os.path.join(MAN, t2.iloc[0]["File Location"].lstrip(".\\"))).items()})
        adc = st[st["Series Description"].str.contains("ADC|Apparent", case=False, na=False)]
        if not adc.empty: o.update({f"adc_{k}": v for k, v in head(os.path.join(MAN, adc.iloc[0]["File Location"].lstrip(".\\"))).items()})
    except Exception as e:
        o["error"] = str(e)[:80]
    rows.append(o)
R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
print("sessions:", len(R), f"| {time.time()-t0:.0f}s")
for c in ("t2_manufacturer", "t2_model", "t2_field", "t2_coil", "t2_coil_siemens", "adc_coil", "adc_coil_siemens", "t2_tx_coil"):
    if c in R: print("\n", c, "(top 14):\n", R[c].fillna("").value_counts().head(14).to_string())

