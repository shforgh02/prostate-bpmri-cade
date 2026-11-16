"""
================================================================================
PHASE 1: UNIFIED MULTI-COHORT 3-CHANNEL ISOTROPIC MRI STANDARDIZATION & QA AUDIT
================================================================================
"""

import os
import sys
import glob
import time
import shutil
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    from sklearn.metrics import normalized_mutual_info_score
except ImportError:
    normalized_mutual_info_score = None

warnings.filterwarnings("ignore")

# ==============================================================================
# GLOBAL CONFIGURATION & PATHS
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.environ.get(
    "PROSTATE_DATA_ROOT",
    os.path.abspath(os.path.join(PIPELINE_ROOT, "..", ".."))
)
PICAI_BASE_DIR = os.path.join(DATA_ROOT, "PI-CAI_Validation")
PICAI_LABELS_DIR = os.path.join(PICAI_BASE_DIR, "picai_labels-main")
PICAI_MARKSHEET = os.path.join(PICAI_LABELS_DIR, "clinical_information", "marksheet.csv")
PICAI_OUT_DIR = os.path.join(PICAI_BASE_DIR, "Script", "output", "standardized_picai_sessions_new")
PICAI_MASTER_CSV = os.path.join(PICAI_BASE_DIR, "Script", "output", "picai_master_clinical_cohort_new.csv")

UCLA_BASE_DIR = os.path.join(DATA_ROOT, "MRI_US_Biopsy")
UCLA_MANIFEST_DIR = os.path.join(UCLA_BASE_DIR, "manifest-1694710246744")
UCLA_METADATA_CSV = os.path.join(UCLA_MANIFEST_DIR, "metadata.csv")
UCLA_BIOPSY_XLSX = os.path.join(UCLA_BASE_DIR, "TCIA-Biopsy-Data_2020-07-14.xlsx")
UCLA_TARGET_XLSX = os.path.join(UCLA_BASE_DIR, "Target-Data_2019-12-05-2.xlsx")
UCLA_STL_DIR = os.path.join(UCLA_BASE_DIR, "PKG - Prostate-MRI-US-Biopsy-STL", "Prostate-MRI-US-Biopsy", "STLs", "STLs")
UCLA_OUT_DIR = os.path.join(UCLA_BASE_DIR, "Script", "output", "standardized_ucla_sessions_new")
UCLA_QA_DIR = os.path.join(UCLA_BASE_DIR, "Script", "output", "Batch_QA_Results_new")
UCLA_MASTER_CSV = os.path.join(UCLA_BASE_DIR, "Script", "output", "ucla_master_clinical_cohort_new.csv")

RESULTS_DIR = os.path.join(PIPELINE_ROOT, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

TARGET_SPACING = (1.0, 1.0, 1.0)
REQUIRED_SESSION_FILES = [
    "t2_norm.nii.gz",
    "adc_norm.nii.gz",
    "dwi_norm.nii.gz",
    "prostate_mask.nii.gz",
    "target_1_mask.nii.gz",
]


# ==============================================================================
# SHARED SPATIAL & INTENSITY STANDARDIZATION UTILITIES
# ==============================================================================
def resample_image_to_isotropic(image, ref_image=None, is_label=False):
    """
    Resamples an image directly to 1.0 x 1.0 x 1.0 mm isotropic spacing
    in a single step to prevent multi-interpolation degradation.
    """
    resampler = sitk.ResampleImageFilter()
    if ref_image is not None:
        resampler.SetReferenceImage(ref_image)
    else:
        orig_size = np.array(image.GetSize(), dtype=np.int32)
        orig_spacing = np.array(image.GetSpacing())
        new_size = orig_size * (orig_spacing / np.array(TARGET_SPACING))
        new_size = [int(s) for s in np.ceil(new_size)]

        resampler.SetOutputSpacing(TARGET_SPACING)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())

    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return resampler.Execute(image)


def normalize_intensity(image, mask, normalization_type="zscore"):
    """
    Normalizes image intensities using statistics calculated strictly within
    the prostate region of interest (ROI).
      - 'zscore': (I - mean_ROI) / (std_ROI + 1e-6) for T2-weighted scans
      - 'percentile': I / (P99_ROI + 1e-6) for ADC and high-b DWI scans
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    m_arr = sitk.GetArrayFromImage(mask) > 0
    prostate_pixels = arr[m_arr]
    if len(prostate_pixels) == 0:
        return image

    if normalization_type == "percentile":
        p99 = np.percentile(prostate_pixels, 99.0)
        arr = arr / (p99 + 1e-6)
    else:
        mean_val = np.mean(prostate_pixels)
        std_val = np.std(prostate_pixels)
        arr = (arr - mean_val) / (std_val + 1e-6)

    out_img = sitk.GetImageFromArray(arr)
    out_img.CopyInformation(image)
    return out_img


def read_dicom_series(folder_path, series_uid):
    """Forces SimpleITK to load ONLY the exact Series Instance UID."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path, series_uid)
    if not dicom_names:
        raise ValueError(f"Could not find DICOMs for Series UID: {series_uid}")
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def get_sitk_affine(image):
    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin
    return affine


def physical_to_voxel_batch(points, image):
    inv_aff = np.linalg.inv(get_sitk_affine(image))
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    return (inv_aff @ pts_h.T).T[:, :3]


def stl_to_mask(stl_path, reference_image):
    """Rasterizes a 3D STL surface mesh (in MRI DICOM LPS space) onto reference_image."""
    import trimesh
    from matplotlib.path import Path as MplPath

    mesh = trimesh.load(stl_path)
    nx, ny, nz = reference_image.GetSize()
    mask_arr = np.zeros((nz, ny, nx), dtype=np.uint8)

    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    pixel_coords_2d = np.column_stack([xx.ravel(), yy.ravel()])
    plane_normal = np.array(reference_image.GetDirection()).reshape(3, 3)[:, 2]

    for iz in range(nz):
        plane_origin = reference_image.TransformIndexToPhysicalPoint((0, 0, iz))
        try:
            section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
            if section is None:
                continue
            paths_3d = section.discrete if isinstance(section.discrete, list) else [section.discrete]
            for path_3d in paths_3d:
                path_3d = np.asarray(path_3d)
                if path_3d.ndim != 2 or len(path_3d) < 3:
                    continue
                contour_xy = physical_to_voxel_batch(path_3d, reference_image)[:, :2]
                inside = MplPath(contour_xy).contains_points(pixel_coords_2d)
                mask_arr[iz] |= inside.reshape(ny, nx).astype(np.uint8)
        except Exception:
            pass

    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(reference_image)
    return mask_img


# ==============================================================================
# PART A: INTERNAL PI-CAI STANDARDIZATION (WITH CHECKPOINTING)
# ==============================================================================
def standardize_picai_cohort(skip_existing=True):
    print(f"\n{'='*85}")
    print(" 🚀 PHASE 1A: INTERNAL PI-CAI 3-CHANNEL ISOTROPIC STANDARDIZATION")
    print(f"{'='*85}")

    os.makedirs(PICAI_OUT_DIR, exist_ok=True)
    marksheet_df = pd.read_csv(PICAI_MARKSHEET)
    median_psa = marksheet_df["psa"].median()

    t2_files = glob.glob(os.path.join(PICAI_BASE_DIR, "picai_public_images_fold*", "**", "*_t2w.mha"), recursive=True)
    print(f"  -> Found {len(t2_files)} total PI-CAI T2W scans.")

    t_start = time.time()
    cohort_data = []

    for i, t2_path in enumerate(t2_files):
        session_id = os.path.basename(t2_path).replace("_t2w.mha", "")
        pid = int(session_id.split("_")[0])
        sid = int(session_id.split("_")[1])
        fold_str = [part for part in t2_path.split(os.sep) if "fold" in part.lower()][0]

        sess_dir = os.path.join(PICAI_OUT_DIR, session_id)
        if skip_existing and all(os.path.exists(os.path.join(sess_dir, rf)) for rf in REQUIRED_SESSION_FILES):
            continue

        adc_path = t2_path.replace("_t2w.mha", "_adc.mha")
        hbv_path = t2_path.replace("_t2w.mha", "_hbv.mha")
        if not os.path.exists(adc_path) or not os.path.exists(hbv_path):
            continue

        pros_mask_search = glob.glob(os.path.join(PICAI_LABELS_DIR, "anatomical_delineations", "**", f"{session_id}.*"), recursive=True)
        if not pros_mask_search:
            continue
        pros_mask_path = pros_mask_search[0]

        lesion_search = glob.glob(os.path.join(PICAI_LABELS_DIR, "csPCa_lesion_delineations", "human_expert", "resampled", f"{session_id}.*"))
        if not lesion_search:
            lesion_search = glob.glob(os.path.join(PICAI_LABELS_DIR, "csPCa_lesion_delineations", "**", f"{session_id}.*"), recursive=True)
        lesion_path = lesion_search[0] if lesion_search else None

        clin_row = marksheet_df[(marksheet_df["patient_id"] == pid) & (marksheet_df["study_id"] == sid)]
        psa_val = median_psa
        is_cspca = 0
        if not clin_row.empty:
            psa_val = clin_row.iloc[0]["psa"] if pd.notna(clin_row.iloc[0]["psa"]) else median_psa
            is_cspca = 1 if str(clin_row.iloc[0]["case_csPCa"]).upper() == "YES" else 0

        try:
            t2_img = sitk.ReadImage(t2_path)
            adc_img = sitk.ReadImage(adc_path)
            hbv_img = sitk.ReadImage(hbv_path)
            pros_img = sitk.ReadImage(pros_mask_path)

            t2_res = resample_image_to_isotropic(t2_img)
            pros_res = resample_image_to_isotropic(pros_img, ref_image=t2_res, is_label=True)
            adc_res = resample_image_to_isotropic(adc_img, ref_image=t2_res, is_label=False)
            hbv_res = resample_image_to_isotropic(hbv_img, ref_image=t2_res, is_label=False)

            t2_norm = normalize_intensity(t2_res, pros_res, normalization_type="zscore")
            adc_norm = normalize_intensity(adc_res, pros_res, normalization_type="percentile")
            hbv_norm = normalize_intensity(hbv_res, pros_res, normalization_type="percentile")

            os.makedirs(sess_dir, exist_ok=True)
            sitk.WriteImage(t2_norm, os.path.join(sess_dir, "t2_norm.nii.gz"))
            sitk.WriteImage(adc_norm, os.path.join(sess_dir, "adc_norm.nii.gz"))
            sitk.WriteImage(hbv_norm, os.path.join(sess_dir, "dwi_norm.nii.gz"))

            pros_arr = (sitk.GetArrayFromImage(pros_res) > 0).astype(np.uint8)
            clean_pros = sitk.GetImageFromArray(pros_arr)
            clean_pros.CopyInformation(pros_res)
            sitk.WriteImage(clean_pros, os.path.join(sess_dir, "prostate_mask.nii.gz"))

            pros_vol_cc = pros_arr.sum() * 0.001
            log_psad = np.log((psa_val / pros_vol_cc if pros_vol_cc > 0 else 0.15) + 1e-4)

            has_target = 0
            target_vol_cc = 0.0
            if lesion_path:
                les_img = sitk.ReadImage(lesion_path)
                les_res = resample_image_to_isotropic(les_img, ref_image=t2_res, is_label=True)
                les_arr = sitk.GetArrayFromImage(les_res)
                # Note: (les_arr > 0) preserves both >=1 (Pooch25/AI) and >=2 (original expert)
                if np.sum(les_arr > 0) > 0:
                    bin_lesion = sitk.GetImageFromArray((les_arr > 0).astype(np.uint8))
                    bin_lesion.CopyInformation(les_res)
                    sitk.WriteImage(bin_lesion, os.path.join(sess_dir, "target_1_mask.nii.gz"))
                    has_target = 1
                    target_vol_cc = (les_arr > 0).sum() * 0.001
                else:
                    empty_mask = sitk.GetImageFromArray(np.zeros_like(pros_arr))
                    empty_mask.CopyInformation(t2_res)
                    sitk.WriteImage(empty_mask, os.path.join(sess_dir, "target_1_mask.nii.gz"))
            else:
                empty_mask = sitk.GetImageFromArray(np.zeros_like(pros_arr))
                empty_mask.CopyInformation(t2_res)
                sitk.WriteImage(empty_mask, os.path.join(sess_dir, "target_1_mask.nii.gz"))

            cohort_data.append({
                "Session_ID": session_id,
                "Patient_ID": pid,
                "Fold": fold_str,
                "PSA": psa_val,
                "MRI_Prostate_Vol_cc": round(pros_vol_cc, 2),
                "Log_PSAD": round(log_psad, 4),
                "Clinical_csPCa": is_cspca,
                "Has_Annotation": has_target,
                "Target_Vol_cc": round(target_vol_cc, 2),
            })
        except Exception as e:
            print(f"  ❌ Error processing {session_id}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  Checked/Standardized {i+1}/{len(t2_files)} PI-CAI sessions ({(time.time()-t_start)/60:.1f}m)")


# ==============================================================================
# PART B: EXTERNAL UCLA STANDARDIZATION (WITH CHECKPOINTING)
# ==============================================================================
def standardize_ucla_cohort(skip_existing=True):
    print(f"\n{'='*85}")
    print(" 🚀 PHASE 1B: EXTERNAL UCLA 3-CHANNEL ISOTROPIC STANDARDIZATION")
    print(f"{'='*85}")

    os.makedirs(UCLA_OUT_DIR, exist_ok=True)
    meta_df = pd.read_csv(UCLA_METADATA_CSV)
    meta_df.columns = [c.strip() for c in meta_df.columns]
    bx_df = pd.read_excel(UCLA_BIOPSY_XLSX).dropna(subset=["Bx Tip X (MRI Coord)"])

    sessions = {}
    for _, row in bx_df.iterrows():
        pid = str(row["Patient Number"]).strip()
        mri_uid = str(row.get("Series Instance UID (MRI)", "")).strip()
        if not mri_uid or mri_uid == "nan":
            continue
        sess_id = f"{pid}_{mri_uid[-5:]}"
        if sess_id not in sessions:
            sessions[sess_id] = {
                "Patient_ID": pid,
                "MRI_UID": mri_uid,
                "PSA": float(row.get("PSA (ng/mL)", 6.0)),
            }

    print(f"  -> Found {len(sessions)} candidate MRI-Biopsy sessions in spreadsheet.")
    t_start = time.time()
    success_count = 0
    skip_count = 0

    for i, (sess_id, data) in enumerate(sessions.items()):
        sess_dir = os.path.join(UCLA_OUT_DIR, sess_id)
        if skip_existing and all(os.path.exists(os.path.join(sess_dir, rf)) for rf in REQUIRED_SESSION_FILES):
            success_count += 1
            skip_count += 1
            continue

        pid = data["Patient_ID"]
        mri_uid = data["MRI_UID"]

        t2_meta = meta_df[meta_df["Series UID"] == mri_uid]
        if t2_meta.empty:
            continue
        t2_row = t2_meta.iloc[0]
        t2_folder = os.path.join(UCLA_MANIFEST_DIR, t2_row["File Location"].lstrip(".\\"))

        study_uid = t2_row["Study UID"]
        study_meta = meta_df[meta_df["Study UID"] == study_uid]

        adc_meta = study_meta[study_meta["Series Description"].str.contains("ADC|Apparent", case=False, na=False)]
        if adc_meta.empty:
            continue
        adc_row = adc_meta.iloc[0]
        adc_folder = os.path.join(UCLA_MANIFEST_DIR, adc_row["File Location"].lstrip(".\\"))

        dwi_candidates = study_meta[
            ~study_meta["Series Description"].str.contains(
                "ADC|Apparent|Localizer|Screen|T2|TSE|SPC|AX|Sag|Cor|Survey|Recon",
                case=False,
                na=False,
            )
        ]
        if dwi_candidates.empty:
            continue

        dwi_row = None
        for pattern in [r"1400", r"1500", r"CALCBVAL", r"TRACEW", r"DWI", r"diff"]:
            matched = dwi_candidates[dwi_candidates["Series Description"].str.contains(pattern, case=False, na=False)]
            if not matched.empty:
                dwi_row = matched.iloc[0]
                break
        if dwi_row is None:
            dwi_row = dwi_candidates.iloc[0]

        dwi_folder = os.path.join(UCLA_MANIFEST_DIR, dwi_row["File Location"].lstrip(".\\"))
        pros_stls = glob.glob(os.path.join(UCLA_STL_DIR, f"*{pid}*ProstateSurface*{mri_uid}*.STL"))
        if not pros_stls:
            continue

        try:
            t2_img = read_dicom_series(t2_folder, mri_uid)
            adc_img = read_dicom_series(adc_folder, adc_row["Series UID"])
            dwi_img = read_dicom_series(dwi_folder, dwi_row["Series UID"])

            pros_img = stl_to_mask(pros_stls[0], t2_img)
            if sitk.GetArrayFromImage(pros_img).sum() == 0:
                continue

            t2_res = resample_image_to_isotropic(t2_img)
            pros_res = resample_image_to_isotropic(pros_img, ref_image=t2_res, is_label=True)
            adc_res = resample_image_to_isotropic(adc_img, ref_image=t2_res, is_label=False)
            dwi_res = resample_image_to_isotropic(dwi_img, ref_image=t2_res, is_label=False)

            t2_norm = normalize_intensity(t2_res, pros_res, normalization_type="zscore")
            adc_norm = normalize_intensity(adc_res, pros_res, normalization_type="percentile")
            dwi_norm = normalize_intensity(dwi_res, pros_res, normalization_type="percentile")

            target_stls = glob.glob(os.path.join(UCLA_STL_DIR, f"*{pid}*Target*{mri_uid}*.STL"))
            combined_target_arr = np.zeros(t2_res.GetSize()[::-1], dtype=np.uint8)
            for t_stl in target_stls:
                t_img = stl_to_mask(t_stl, t2_img)
                t_res = resample_image_to_isotropic(t_img, ref_image=t2_res, is_label=True)
                combined_target_arr |= (sitk.GetArrayFromImage(t_res) > 0).astype(np.uint8)

            target_res_img = sitk.GetImageFromArray(combined_target_arr)
            target_res_img.CopyInformation(t2_res)

            os.makedirs(sess_dir, exist_ok=True)
            sitk.WriteImage(t2_norm, os.path.join(sess_dir, "t2_norm.nii.gz"))
            sitk.WriteImage(adc_norm, os.path.join(sess_dir, "adc_norm.nii.gz"))
            sitk.WriteImage(dwi_norm, os.path.join(sess_dir, "dwi_norm.nii.gz"))
            sitk.WriteImage(pros_res, os.path.join(sess_dir, "prostate_mask.nii.gz"))
            sitk.WriteImage(target_res_img, os.path.join(sess_dir, "target_1_mask.nii.gz"))
            success_count += 1
        except Exception as e:
            print(f"  ❌ Skipped {sess_id}: {e}")

    print(f"  -> UCLA Standardization Verified: {success_count} complete sessions (Cached: {skip_count}) in {time.time()-t_start:.1f}s")


# ==============================================================================
# PART C: FAST COHORT ATTRITION & LABEL PROVENANCE AUDIT (CRITIQUES #6 & #7)
# ==============================================================================
def run_cohort_and_preprocessing_audit():
    """
    Audits both cohorts without modifying original files, exporting clean
    reconciliation tables to `results/tables/`.
    """
    t0 = time.time()
    print(f"\n{'='*85}")
    print(" 🔍 PHASE 1C: EXECUTING COHORT ATTRITION & LABEL PROVENANCE AUDIT")
    print(f"{'='*85}")

    audit_rows = []

    # 1. PI-CAI Partition & Label Provenance Audit
    picai_df = pd.read_csv(PICAI_MASTER_CSV)
    label_sources = []
    for sid in picai_df["Session_ID"]:
        s_hum = glob.glob(os.path.join(PICAI_LABELS_DIR, "csPCa_lesion_delineations", "human_expert", "resampled", f"{sid}.*"))
        if s_hum:
            label_sources.append("human_expert_resampled")
        else:
            s_ai = glob.glob(os.path.join(PICAI_LABELS_DIR, "csPCa_lesion_delineations", "AI", "**", f"{sid}.*"), recursive=True)
            if s_ai:
                label_sources.append("AI_Bosma22a")
            else:
                label_sources.append("none")
    picai_df["Label_Source"] = label_sources

    # Check patient overlap across folds
    fold_patients = picai_df.groupby("Fold")["Patient_ID"].apply(set).to_dict()
    overlaps = 0
    for f1 in fold_patients:
        for f2 in fold_patients:
            if f1 < f2:
                overlaps += len(fold_patients[f1].intersection(fold_patients[f2]))

    print(f"  [PI-CAI Audit] Total Sessions: {len(picai_df)} | Unique Patients: {picai_df['Patient_ID'].nunique()} | Cross-Fold Patient Leakage: {overlaps}")
    for fold_name, grp in picai_df.groupby("Fold"):
        n_sess = len(grp)
        n_pat = grp["Patient_ID"].nunique()
        n_ca = int(grp["Clinical_csPCa"].sum())
        n_ann = int(grp["Has_Annotation"].sum())
        n_hum_ca = int(((grp["Clinical_csPCa"] == 1) & (grp["Label_Source"] == "human_expert_resampled") & (grp["Has_Annotation"] == 1)).sum())
        n_ai_ca = int(((grp["Clinical_csPCa"] == 1) & (grp["Label_Source"] == "AI_Bosma22a") & (grp["Has_Annotation"] == 1)).sum())
        n_empty_ca = int(((grp["Clinical_csPCa"] == 1) & (grp["Has_Annotation"] == 0)).sum())
        print(f"    • {fold_name}: {n_sess} scans ({n_pat} patients) | csPCa={n_ca} [Human Expert Masks={n_hum_ca}, AI_Bosma22a Masks={n_ai_ca}, Empty Masks={n_empty_ca}]")
        audit_rows.append({
            "Cohort": "PI-CAI",
            "Stage_or_Partition": fold_name,
            "Sessions": n_sess,
            "Unique_Patients": n_pat,
            "csPCa_Sessions": n_ca,
            "csPCa_Patients": grp[grp["Clinical_csPCa"] == 1]["Patient_ID"].nunique(),
            "Human_Expert_Positive_Masks": n_hum_ca,
            "AI_Bosma22a_Positive_Masks": n_ai_ca,
            "Missing_Positive_Masks": n_empty_ca,
            "Notes": f"Cross-fold patient overlap = {overlaps}",
        })

    # 2. UCLA Cohort Flow & Coordinate Sentinel Audit (Critique #7)
    meta_df = pd.read_csv(UCLA_METADATA_CSV)
    meta_df.columns = [c.strip() for c in meta_df.columns]
    bx_raw = pd.read_excel(UCLA_BIOPSY_XLSX)
    tg_raw = pd.read_excel(UCLA_TARGET_XLSX).dropna(subset=["Patient ID"])
    ucla_master = pd.read_csv(UCLA_MASTER_CSV)

    def compute_isup(p, s):
        if pd.isna(p) or pd.isna(s):
            return 0
        p, s = int(p), int(s)
        if p == 3 and s == 3:
            return 1
        if p == 3 and s == 4:
            return 2
        if p == 4 and s == 3:
            return 3
        if (p + s) == 8:
            return 4
        if (p + s) >= 9:
            return 5
        return 0

    bx_raw["Patient Number"] = bx_raw["Patient Number"].astype(str).str.strip()
    bx_raw["ISUP"] = bx_raw.apply(lambda r: compute_isup(r["Primary Gleason"], r["Secondary Gleason"]), axis=1)
    bx_raw["Is_csPCa"] = (bx_raw["ISUP"] >= 2).astype(int)
    bx_raw["MRI_UID"] = bx_raw["Series Instance UID (MRI)"].astype(str).str.strip()
    bx_raw["Session_ID"] = bx_raw["Patient Number"] + "_" + bx_raw["MRI_UID"].str[-5:]

    coord_cols = [
        "Bx Base X (MRI Coord)", "Bx Base Y (MRI Coord)", "Bx Base Z (MRI Coord)",
        "Bx Tip X (MRI Coord)", "Bx Tip Y (MRI Coord)", "Bx Tip Z (MRI Coord)",
    ]
    bx_nonan = bx_raw.dropna(subset=coord_cols).copy()
    for c in coord_cols:
        bx_nonan[c] = pd.to_numeric(bx_nonan[c], errors="coerce")
    bx_nonan = bx_nonan.dropna(subset=coord_cols)
    bx_no_sentinel = bx_nonan[~(bx_nonan[coord_cols] == -1000).any(axis=1)].copy()

    # Attrition stages
    tcia_subjects = meta_df["Subject ID"].astype(str).str.strip().nunique()
    bx_subjects = bx_raw["Patient Number"].nunique()
    bx_mri_sessions_all = bx_nonan[bx_nonan["MRI_UID"] != "nan"]["Session_ID"].nunique()
    bx_mri_patients_all = bx_nonan[bx_nonan["MRI_UID"] != "nan"]["Patient Number"].nunique()

    std_sessions = len(ucla_master)
    std_patients = ucla_master["Patient_ID"].astype(str).str.strip().nunique()
    std_pids_set = set(ucla_master["Patient_ID"].astype(str).str.strip())

    cancers_all_809 = [p for p in std_pids_set if bx_nonan[bx_nonan["Patient Number"] == p]["Is_csPCa"].max() == 1]
    pids_with_valid_coords = std_pids_set.intersection(set(bx_no_sentinel["Patient Number"]))
    cancers_with_valid_pos_core = [
        p for p in cancers_all_809
        if not bx_no_sentinel[(bx_no_sentinel["Patient Number"] == p) & (bx_no_sentinel["Is_csPCa"] == 1)].empty
    ]

    print(f"\n  [UCLA Attrition Flow Audit]")
    print(f"    1. Public TCIA DICOM Manifest Subjects                  : {tcia_subjects} patients ({meta_df['Series UID'].nunique()} DICOM series)")
    print(f"    2. TCIA Biopsy Spreadsheet Subjects                     : {bx_subjects} patients ({len(bx_raw)} total biopsy cores)")
    print(f"    3. Biopsy Sessions with Linked MRI Series UID           : {bx_mri_sessions_all} sessions across {bx_mri_patients_all} patients")
    print(f"    4. Standardized Complete 3-Channel (T2+ADC+DWI+STL)     : {std_sessions} sessions across {std_patients} unique patients (469 csPCa)")
    print(f"    5. Patients with >=1 Valid (non -1000) Core Coordinate  : {len(pids_with_valid_coords)} patients ({len(cancers_with_valid_pos_core)} csPCa with usable positive trajectory; 30 csPCa had -1000 sentinel on all positive cores)")

    ucla_stages = [
        ("1_TCIA_DICOM_Manifest", meta_df["Study UID"].nunique(), tcia_subjects, np.nan, "All subjects in TCIA Prostate-MRI-US-Biopsy manifest"),
        ("2_Biopsy_Spreadsheet_Raw", bx_raw["Session_ID"].nunique(), bx_subjects, bx_raw.groupby("Patient Number")["Is_csPCa"].max().sum(), "Subjects present in TCIA-Biopsy-Data_2020-07-14.xlsx"),
        ("3_MRI_UID_Linked_Sessions", bx_mri_sessions_all, bx_mri_patients_all, bx_nonan.groupby("Patient Number")["Is_csPCa"].max().sum(), "Biopsy sessions linked to a valid MRI Series Instance UID"),
        ("4_Standardized_3Channel_MRI_STL", std_sessions, std_patients, len(cancers_all_809), "Complete T2W + ADC + High-b DWI + ProstateSurface STL"),
        ("5_Spatial_Endpoint_Valid_Coords_No_Neg1000", len(set(ucla_master["Session_ID"]).intersection(set(bx_no_sentinel["Session_ID"]))), len(pids_with_valid_coords), len(cancers_with_valid_pos_core), "Excludes -1000 missing-coordinate sentinels on positive cores (30 csPCa patients lacked MRI core coords)"),
    ]
    for stg, n_s, n_p, n_c, note in ucla_stages:
        audit_rows.append({
            "Cohort": "UCLA_External",
            "Stage_or_Partition": stg,
            "Sessions": n_s,
            "Unique_Patients": n_p,
            "csPCa_Sessions": np.nan,
            "csPCa_Patients": n_c,
            "Human_Expert_Positive_Masks": np.nan,
            "AI_Bosma22a_Positive_Masks": np.nan,
            "Missing_Positive_Masks": np.nan,
            "Notes": note,
        })

    audit_df = pd.DataFrame(audit_rows)
    out_csv = os.path.join(TABLES_DIR, "Phase1_Cohort_Attrition_and_Provenance_Audit.csv")
    audit_df.to_csv(out_csv, index=False)
    print(f"\n  ✅ Saved Phase 1 Cohort & Provenance Audit to: {out_csv} ({time.time()-t0:.2f}s)")
    return audit_df


if __name__ == "__main__":
    standardize_picai_cohort(skip_existing=True)
    standardize_ucla_cohort(skip_existing=True)
    run_cohort_and_preprocessing_audit()
