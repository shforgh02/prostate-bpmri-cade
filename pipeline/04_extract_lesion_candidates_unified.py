import os
import sys
import time
import gzip
import struct
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import label, center_of_mass, binary_dilation
from report_guided_annotation import extract_lesion_candidates

NIFTI_DTYPE_MAP = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}

def load_nifti_fast(path):
    """
    Fast zero-dependency NIfTI-1 (.nii.gz) reader returning (data_xyz, affine_ras_4x4).
    Verified bit-for-bit identical to nibabel.load().
    """
    with gzip.open(path, "rb") as f:
        raw = f.read()
    dims = struct.unpack("<8h", raw[40:56])
    nx, ny, nz = int(dims[1]), int(dims[2]), int(dims[3])
    dt_code = struct.unpack("<h", raw[70:72])[0]
    vox_offset = int(struct.unpack("<f", raw[108:112])[0])
    scl_slope, scl_inter = struct.unpack("<ff", raw[112:120])
    sform_code = struct.unpack("<h", raw[254:256])[0]

    if sform_code > 0:
        srow_x = struct.unpack("<4f", raw[280:296])
        srow_y = struct.unpack("<4f", raw[296:312])
        srow_z = struct.unpack("<4f", raw[312:328])
        affine = np.array([srow_x, srow_y, srow_z, [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        pixdim = struct.unpack("<8f", raw[76:108])
        qfac = -1.0 if pixdim[0] < 0 else 1.0
        dx, dy, dz = float(pixdim[1]), float(pixdim[2]), float(pixdim[3]) * qfac
        qb, qc, qd, qx, qy, qz = struct.unpack("<6f", raw[256:280])
        qa_sq = max(0.0, 1.0 - (qb * qb + qc * qc + qd * qd))
        qa = np.sqrt(qa_sq)
        r = np.array([
            [qa*qa + qb*qb - qc*qc - qd*qd, 2*(qb*qc - qa*qd), 2*(qb*qd + qa*qc)],
            [2*(qb*qc + qa*qd), qa*qa + qc*qc - qb*qb - qd*qd, 2*(qc*qd - qa*qb)],
            [2*(qb*qd - qa*qc), 2*(qc*qd + qa*qb), qa*qa + qd*qd - qb*qb - qc*qc],
        ], dtype=np.float64)
        r = r @ np.diag([dx, dy, dz])
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = r
        affine[:3, 3] = [qx, qy, qz]

    dtype = NIFTI_DTYPE_MAP.get(dt_code, np.float32)
    n_vox = nx * ny * nz
    arr = np.frombuffer(raw, dtype=dtype, count=n_vox, offset=vox_offset).reshape((nx, ny, nz), order="F")
    if scl_slope != 0.0 and not np.isnan(scl_slope) and (scl_slope != 1.0 or scl_inter != 0.0):
        arr = arr.astype(np.float32) * scl_slope + scl_inter
    return arr, affine


PROD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get(
    "PROSTATE_DATA_ROOT",
    os.path.abspath(os.path.join(PROD_DIR, "..", ".."))
)
AUDIT_DIR = os.path.join(PROD_DIR, "results", "tables")
os.makedirs(AUDIT_DIR, exist_ok=True)
def find_first_existing(path_list):
    for p in path_list:
        if os.path.exists(p):
            return p
    return path_list[0]

SEG_PROBMAP_DIR = find_first_existing([
    os.path.join(PROD_DIR, "results", "segresnet_probmaps"),
    os.path.join(DATA_ROOT, "Final Script", "antigravity", "results", "segresnet_probmaps"),
])
NNU_PRED_DIR = find_first_existing([
    os.path.join(DATA_ROOT, "MRI_US_Biopsy", "Script", "nnUNet_predictions", "predictions"),
])
SESSIONS_DIR = find_first_existing([
    os.path.join(DATA_ROOT, "MRI_US_Biopsy", "Script", "output", "standardized_ucla_sessions_new"),
])

CORES_REF_CSV = os.path.join(AUDIT_DIR, "CORES_REFERENCE.csv")
MANIFEST_CSV = os.path.join(AUDIT_DIR, "ANALYSIS_MANIFEST.csv")

def vox_to_lps(vox_xyz, affine_ras):
    """Converts voxel coordinates to physical DICOM LPS mm."""
    vox_arr = np.asarray(vox_xyz, dtype=np.float64)
    single = (vox_arr.ndim == 1)
    if single:
        vox_arr = vox_arr[None, :]
    ras = (affine_ras[:3, :3] @ vox_arr.T).T + affine_ras[:3, 3]
    lps = ras * np.array([-1.0, -1.0, 1.0])
    return lps[0] if single else lps

def point_to_segments_dist(pt_lps, bases_lps, tips_lps):
    """Vectorized clamped distance from 3D point to M segments."""
    if len(bases_lps) == 0:
        return np.inf
    ab = tips_lps - bases_lps
    ap = pt_lps[None, :] - bases_lps
    ab_len2 = np.sum(ab * ab, axis=1)
    t = np.where(ab_len2 > 1e-12, np.sum(ap * ab, axis=1) / ab_len2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = bases_lps + t[:, None] * ab
    dists = np.linalg.norm(pt_lps[None, :] - proj, axis=1)
    return float(np.min(dists))

def surface_vox_to_segments_dist(comp_mask_3d, comp_vox_xyz, affine_ras, bases_lps, tips_lps, step_mm=0.5):
    """Volumetric surface distance from segment to component."""
    if len(bases_lps) == 0 or len(comp_vox_xyz) == 0:
        return np.inf
    inv_affine = np.linalg.inv(affine_ras)
    shape = comp_mask_3d.shape
    min_dist = np.inf

    for i in range(len(bases_lps)):
        b_lps = bases_lps[i]
        t_lps = tips_lps[i]
        b_ras = b_lps * np.array([-1.0, -1.0, 1.0])
        t_ras = t_lps * np.array([-1.0, -1.0, 1.0])
        vec = t_ras - b_ras
        length = float(np.linalg.norm(vec))
        n_pts = max(2, int(np.ceil(length / step_mm)) + 1)
        s_ras = b_ras[None, :] + np.linspace(0.0, 1.0, n_pts)[:, None] * vec[None, :]
        vox_f = (inv_affine[:3, :3] @ s_ras.T).T + inv_affine[:3, 3]
        vox_i = np.round(vox_f).astype(np.int32)
        in_b = (
            (vox_i[:, 0] >= 0) & (vox_i[:, 0] < shape[0]) &
            (vox_i[:, 1] >= 0) & (vox_i[:, 1] < shape[1]) &
            (vox_i[:, 2] >= 0) & (vox_i[:, 2] < shape[2])
        )
        if np.any(in_b):
            v_in = vox_i[in_b]
            if np.any(comp_mask_3d[v_in[:, 0], v_in[:, 1], v_in[:, 2]]):
                return 0.0  # Direct puncture

        # Sample points on segment in LPS
        s_lps = s_ras * np.array([-1.0, -1.0, 1.0])
        # Comp points in LPS (subsample if huge)
        comp_lps = vox_to_lps(comp_vox_xyz[::3] if len(comp_vox_xyz) > 3000 else comp_vox_xyz, affine_ras)
        d2 = np.min(np.sum((s_lps[:, None, :] - comp_lps[None, :, :]) ** 2, axis=2))
        d = np.sqrt(d2)
        if d < min_dist:
            min_dist = d
    return float(min_dist)

def extract_custom_dynamic_blobs(prob_xyz, max_cands=5, min_prob=0.01, min_vol_vox=30):
    """Legacy custom dynamic peak-seeded lesion candidate extractor (sensitivity analysis)."""
    work = prob_xyz.astype(np.float32).copy()
    sx, sy, sz = work.shape
    blobs = []
    for _ in range(max_cands * 3):
        if len(blobs) >= max_cands:
            break
        flat_idx = int(np.argmax(work))
        p_max = float(work.flat[flat_idx])
        if p_max < min_prob:
            break
        px, py, pz = np.unravel_index(flat_idx, work.shape)
        x0, x1 = max(0, px - 35), min(sx, px + 36)
        y0, y1 = max(0, py - 35), min(sy, py + 36)
        z0, z1 = max(0, pz - 35), min(sz, pz + 36)
        sub_w = work[x0:x1, y0:y1, z0:z1]
        lpx, lpy, lpz = px - x0, py - y0, pz - z0

        t_dyn = max(min_prob, 0.45 * p_max)
        lbl_box, _ = label(sub_w >= t_dyn)
        c_id = int(lbl_box[lpx, lpy, lpz])
        sub_m = (lbl_box == c_id) if c_id > 0 else np.zeros_like(sub_w, dtype=bool)
        sub_m[lpx, lpy, lpz] = True

        if int(np.sum(sub_m)) > 25000:
            t_dyn = max(min_prob, 0.75 * p_max)
            lbl_box, _ = label(sub_w >= t_dyn)
            c_id = int(lbl_box[lpx, lpy, lpz])
            sub_m = (lbl_box == c_id) if c_id > 0 else np.zeros_like(sub_w, dtype=bool)
            sub_m[lpx, lpy, lpz] = True

        n_vox = int(np.sum(sub_m))
        v_loc = np.argwhere(sub_m)
        offset = np.array([x0, y0, z0])
        v_glob = v_loc + offset

        # Zero out extracted component + 5mm NMS neighborhood
        sub_w[sub_m] = 0.0
        nx0, nx1 = max(0, px - 5), min(sx, px + 6)
        ny0, ny1 = max(0, py - 5), min(sy, py + 6)
        nz0, nz1 = max(0, pz - 5), min(sz, pz + 6)
        work[nx0:nx1, ny0:ny1, nz0:nz1] = 0.0

        if n_vox < min_vol_vox:
            continue

        com_vox = np.mean(v_glob, axis=0)
        peak_vox = np.array([float(px), float(py), float(pz)])
        comp_3d = np.zeros(work.shape, dtype=bool)
        comp_3d[v_glob[:, 0], v_glob[:, 1], v_glob[:, 2]] = True

        blobs.append({
            "p_max": p_max,
            "n_vox": n_vox,
            "vol_cc": n_vox * 0.001,
            "com_vox": com_vox,
            "peak_vox": peak_vox,
            "mask_c": comp_3d,
            "vox_idx": v_glob,
        })
    return blobs

def run_unified_candidate_extraction():
    t0 = time.time()
    print("=" * 85)
    print("RUNNING UNIFIED CANDIDATE EXTRACTION (FIXED CC, OFFICIAL RGA & CUSTOM)")
    print("=" * 85)

    manifest_df = pd.read_csv(MANIFEST_CSV)
    cores_df = pd.read_csv(CORES_REF_CSV)

    # Pre-index positive and control cores per session
    sess_cores_map = {}
    for sid, grp in cores_df.groupby("Session_ID_Matched"):
        # Valid csPCa cores
        cspca_grp = grp[grp["Is_Valid_csPCa_Trajectory"] == 1]
        gg1_grp = grp[(grp["Is_GG1_Indolent"] == 1) & (grp["Has_Valid_Physical_Trajectory"] == 1)]
        ben_grp = grp[(grp["Pathology_Category"] == "benign") & (grp["Has_Valid_Physical_Trajectory"] == 1)]
        unk_grp = grp[(grp["Pathology_Category"] == "cancer_missing_gleason") & (grp["Has_Valid_Physical_Trajectory"] == 1)]

        def get_segments(sub_df):
            if len(sub_df) == 0:
                return np.zeros((0, 3)), np.zeros((0, 3))
            b = sub_df[["Bx Base X (MRI Coord)", "Bx Base Y (MRI Coord)", "Bx Base Z (MRI Coord)"]].values
            t = sub_df[["Bx Tip X (MRI Coord)", "Bx Tip Y (MRI Coord)", "Bx Tip Z (MRI Coord)"]].values
            return b, t

        sess_cores_map[sid] = {
            "cspca": get_segments(cspca_grp),
            "gg1": get_segments(gg1_grp),
            "benign": get_segments(ben_grp),
            "unknown": get_segments(unk_grp),
            "has_valid_cspca": len(cspca_grp) > 0,
        }

    all_candidates = []
    
    # Process each session in manifest
    n_total = len(manifest_df)
    for idx, row in manifest_df.iterrows():
        sid = row["Session_ID"]
        pid = row["Patient_ID"]
        is_canon = row["Is_Canonical_Patient_Session"]
        has_seg = row["Has_SegResNet_Inference"]
        has_nnu = row["Has_nnUNet_Inference"]

        core_info = sess_cores_map.get(sid, {
            "cspca": (np.zeros((0, 3)), np.zeros((0, 3))),
            "gg1": (np.zeros((0, 3)), np.zeros((0, 3))),
            "benign": (np.zeros((0, 3)), np.zeros((0, 3))),
            "unknown": (np.zeros((0, 3)), np.zeros((0, 3))),
            "has_valid_cspca": False,
        })
        cs_b, cs_t = core_info["cspca"]
        gg1_b, gg1_t = core_info["gg1"]
        ben_b, ben_t = core_info["benign"]
        unk_b, unk_t = core_info["unknown"]

        # Models to extract
        models_data = []
        if has_seg:
            seg_p = os.path.join(SEG_PROBMAP_DIR, f"{sid}.npz")
            if os.path.exists(seg_p):
                sd = np.load(seg_p)
                models_data.append(("3D_SegResNet", sd["heatmap"].astype(np.float32), sd["affine"], sd["pros_mask"]))

        if has_nnu:
            nnu_p = os.path.join(NNU_PRED_DIR, f"UCLA_{sid}.npz")
            pkl_p = os.path.join(NNU_PRED_DIR, f"UCLA_{sid}.pkl")
            if os.path.exists(nnu_p) and os.path.exists(pkl_p):
                pros_file = os.path.join(SESSIONS_DIR, sid, "prostate_mask.nii.gz")
                if os.path.exists(pros_file):
                    pros_arr_raw, nnu_aff = load_nifti_fast(pros_file)
                    pros_arr = (pros_arr_raw > 0)
                    dil_pros = binary_dilation(pros_arr, iterations=4)
                    nd = np.load(nnu_p)
                    raw_sm = nd["softmax"][1].astype(np.float32)  # (Z, Y, X)
                    with open(pkl_p, "rb") as pf:
                        nnu_meta = pickle.load(pf)

                    target_zyx = dil_pros.shape[::-1]
                    if raw_sm.shape != target_zyx:
                        full_zyx = np.zeros(target_zyx, dtype=np.float32)
                        bbox = nnu_meta.get("crop_bbox")
                        if bbox is not None:
                            full_zyx[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = raw_sm
                        else:
                            sz = [min(full_zyx.shape[i], raw_sm.shape[i]) for i in range(3)]
                            full_zyx[:sz[0], :sz[1], :sz[2]] = raw_sm[:sz[0], :sz[1], :sz[2]]
                        sm_xyz = full_zyx.transpose(2, 1, 0)
                    else:
                        sm_xyz = raw_sm.transpose(2, 1, 0)

                    sm_xyz[~dil_pros] = 0.0
                    models_data.append(("Pretrained_nnUNet", sm_xyz, nnu_aff, pros_arr))

        for model_name, prob_map, affine_ras, pros_mask in models_data:
            # EXTRACTOR 1: Shared Fixed Connected Components at prob >= 0.10
            lbl_arr, num_feats = label(prob_map >= 0.10)
            fixed_cands = []
            for c_id in range(1, num_feats + 1):
                mask_c = (lbl_arr == c_id)
                n_vox = int(mask_c.sum())
                vol_cc = n_vox * 0.001
                if vol_cc < 0.03 or vol_cc > 25.0:
                    continue
                vox_idx = np.argwhere(mask_c)
                com_vox = np.mean(vox_idx, axis=0)
                # True argmax peak voxel within component
                c_probs = prob_map[mask_c]
                max_prob = float(np.max(c_probs))
                mean_prob = float(np.mean(c_probs))
                flat_argmax = np.argmax(prob_map * mask_c)
                peak_vox = np.unravel_index(flat_argmax, prob_map.shape)

                fixed_cands.append({
                    "c_id": c_id,
                    "max_prob": max_prob,
                    "mean_prob": mean_prob,
                    "vol_cc": vol_cc,
                    "n_vox": n_vox,
                    "com_vox": com_vox,
                    "peak_vox": peak_vox,
                    "mask_c": mask_c,
                    "vox_idx": vox_idx,
                })
            # Deterministic sorting: Max_Prob desc, then vol_cc desc, then c_id
            fixed_cands.sort(key=lambda x: (-x["max_prob"], -x["vol_cc"], x["c_id"]))

            for rank, fc in enumerate(fixed_cands[:5], start=1):
                com_lps = vox_to_lps(fc["com_vox"], affine_ras)
                peak_lps = vox_to_lps(fc["peak_vox"], affine_ras)
                offset_mm = float(np.linalg.norm(com_lps - peak_lps))

                com_d_cs = point_to_segments_dist(com_lps, cs_b, cs_t)
                peak_d_cs = point_to_segments_dist(peak_lps, cs_b, cs_t)
                surf_d_cs = surface_vox_to_segments_dist(fc["mask_c"], fc["vox_idx"], affine_ras, cs_b, cs_t)

                com_d_gg1 = point_to_segments_dist(com_lps, gg1_b, gg1_t)
                com_d_ben = point_to_segments_dist(com_lps, ben_b, ben_t)
                com_d_unk = point_to_segments_dist(com_lps, unk_b, unk_t)

                # Classify Candidate State (10mm tolerance)
                if com_d_cs <= 10.0:
                    cand_state = "csPCa_Correspondence"
                elif com_d_gg1 <= 10.0:
                    cand_state = "Indolent_GG1_Correspondence"
                elif com_d_ben <= 10.0:
                    cand_state = "Benign_Only_Correspondence"
                elif com_d_unk <= 10.0:
                    cand_state = "Unknown_Grade_Cancer_Proximity"
                else:
                    cand_state = "No_Recorded_Core_Correspondence"

                all_candidates.append({
                    "Candidate_UID": f"{model_name}_FixedCC_{sid}_cand{rank}",
                    "Model": model_name,
                    "Extractor": "Shared_Fixed_CC_0.10",
                    "Session_ID": sid,
                    "Patient_ID": pid,
                    "Is_Canonical": is_canon,
                    "Candidate_Rank": rank,
                    "Max_Prob": fc["max_prob"],
                    "Mean_Prob": fc["mean_prob"],
                    "Volume_cc": fc["vol_cc"],
                    "Voxel_Count": fc["n_vox"],
                    "CoM_LPS_X_mm": round(com_lps[0], 3),
                    "CoM_LPS_Y_mm": round(com_lps[1], 3),
                    "CoM_LPS_Z_mm": round(com_lps[2], 3),
                    "Peak_LPS_X_mm": round(peak_lps[0], 3),
                    "Peak_LPS_Y_mm": round(peak_lps[1], 3),
                    "Peak_LPS_Z_mm": round(peak_lps[2], 3),
                    "CoM_Peak_Offset_mm": round(offset_mm, 3),
                    "CoM_Dist_csPCa_mm": round(com_d_cs, 3) if np.isfinite(com_d_cs) else np.nan,
                    "Peak_Dist_csPCa_mm": round(peak_d_cs, 3) if np.isfinite(peak_d_cs) else np.nan,
                    "Surf3D_Dist_csPCa_mm": round(surf_d_cs, 3) if np.isfinite(surf_d_cs) else np.nan,
                    "CoM_Dist_GG1_mm": round(com_d_gg1, 3) if np.isfinite(com_d_gg1) else np.nan,
                    "CoM_Dist_Benign_mm": round(com_d_ben, 3) if np.isfinite(com_d_ben) else np.nan,
                    "Candidate_State": cand_state,
                })

            # EXTRACTOR 2: Official Report-Guided-Annotation (RGA) Dynamic Extractor
            # Run official RGA extractor on prob_map
            hard_blobs, confidences, blobs_idx = extract_lesion_candidates(
                prob_map.astype(np.float64),
                threshold="dynamic-fast",
                num_lesions_to_extract=5,
                min_voxels_detection=30,
                dynamic_threshold_factor=2.5,
            )
            rga_cands = []
            for l_entry in confidences:
                l_id, conf = l_entry
                c_mask = (blobs_idx == l_id)
                n_v = int(c_mask.sum())
                if n_v < 30:
                    continue
                v_cc = n_v * 0.001
                v_idx = np.argwhere(c_mask)
                com_v = np.mean(v_idx, axis=0)
                fl_arg = np.argmax(prob_map * c_mask)
                peak_v = np.unravel_index(fl_arg, prob_map.shape)
                rga_cands.append({
                    "l_id": l_id,
                    "conf": float(conf),
                    "vol_cc": v_cc,
                    "n_vox": n_v,
                    "com_vox": com_v,
                    "peak_vox": peak_v,
                    "mask_c": c_mask,
                    "vox_idx": v_idx,
                })
            rga_cands.sort(key=lambda x: (-x["conf"], -x["vol_cc"], x["l_id"]))

            for rank, rc in enumerate(rga_cands[:5], start=1):
                com_lps = vox_to_lps(rc["com_vox"], affine_ras)
                peak_lps = vox_to_lps(rc["peak_vox"], affine_ras)
                offset_mm = float(np.linalg.norm(com_lps - peak_lps))

                com_d_cs = point_to_segments_dist(com_lps, cs_b, cs_t)
                peak_d_cs = point_to_segments_dist(peak_lps, cs_b, cs_t)
                surf_d_cs = surface_vox_to_segments_dist(rc["mask_c"], rc["vox_idx"], affine_ras, cs_b, cs_t)

                com_d_gg1 = point_to_segments_dist(com_lps, gg1_b, gg1_t)
                com_d_ben = point_to_segments_dist(com_lps, ben_b, ben_t)
                com_d_unk = point_to_segments_dist(com_lps, unk_b, unk_t)

                if com_d_cs <= 10.0:
                    cand_state = "csPCa_Correspondence"
                elif com_d_gg1 <= 10.0:
                    cand_state = "Indolent_GG1_Correspondence"
                elif com_d_ben <= 10.0:
                    cand_state = "Benign_Only_Correspondence"
                elif com_d_unk <= 10.0:
                    cand_state = "Unknown_Grade_Cancer_Proximity"
                else:
                    cand_state = "No_Recorded_Core_Correspondence"

                all_candidates.append({
                    "Candidate_UID": f"{model_name}_OfficialRGA_{sid}_cand{rank}",
                    "Model": model_name,
                    "Extractor": "Official_RGA_Dynamic",
                    "Session_ID": sid,
                    "Patient_ID": pid,
                    "Is_Canonical": is_canon,
                    "Candidate_Rank": rank,
                    "Max_Prob": rc["conf"],
                    "Mean_Prob": float(np.mean(prob_map[rc["mask_c"]])),
                    "Volume_cc": rc["vol_cc"],
                    "Voxel_Count": rc["n_vox"],
                    "CoM_LPS_X_mm": round(com_lps[0], 3),
                    "CoM_LPS_Y_mm": round(com_lps[1], 3),
                    "CoM_LPS_Z_mm": round(com_lps[2], 3),
                    "Peak_LPS_X_mm": round(peak_lps[0], 3),
                    "Peak_LPS_Y_mm": round(peak_lps[1], 3),
                    "Peak_LPS_Z_mm": round(peak_lps[2], 3),
                    "CoM_Peak_Offset_mm": round(offset_mm, 3),
                    "CoM_Dist_csPCa_mm": round(com_d_cs, 3) if np.isfinite(com_d_cs) else np.nan,
                    "Peak_Dist_csPCa_mm": round(peak_d_cs, 3) if np.isfinite(peak_d_cs) else np.nan,
                    "Surf3D_Dist_csPCa_mm": round(surf_d_cs, 3) if np.isfinite(surf_d_cs) else np.nan,
                    "CoM_Dist_GG1_mm": round(com_d_gg1, 3) if np.isfinite(com_d_gg1) else np.nan,
                    "CoM_Dist_Benign_mm": round(com_d_ben, 3) if np.isfinite(com_d_ben) else np.nan,
                    "Candidate_State": cand_state,
                })

            # EXTRACTOR 3: Legacy Custom Dynamic Extractor (Sensitivity Analysis)
            custom_blobs = extract_custom_dynamic_blobs(prob_map, max_cands=5)
            custom_blobs.sort(key=lambda x: (-x["p_max"], -x["vol_cc"]))
            for rank, cb in enumerate(custom_blobs[:5], start=1):
                com_lps = vox_to_lps(cb["com_vox"], affine_ras)
                peak_lps = vox_to_lps(cb["peak_vox"], affine_ras)
                offset_mm = float(np.linalg.norm(com_lps - peak_lps))

                com_d_cs = point_to_segments_dist(com_lps, cs_b, cs_t)
                peak_d_cs = point_to_segments_dist(peak_lps, cs_b, cs_t)
                surf_d_cs = surface_vox_to_segments_dist(cb["mask_c"], cb["vox_idx"], affine_ras, cs_b, cs_t)

                com_d_gg1 = point_to_segments_dist(com_lps, gg1_b, gg1_t)
                com_d_ben = point_to_segments_dist(com_lps, ben_b, ben_t)
                com_d_unk = point_to_segments_dist(com_lps, unk_b, unk_t)

                if com_d_cs <= 10.0:
                    cand_state = "csPCa_Correspondence"
                elif com_d_gg1 <= 10.0:
                    cand_state = "Indolent_GG1_Correspondence"
                elif com_d_ben <= 10.0:
                    cand_state = "Benign_Only_Correspondence"
                elif com_d_unk <= 10.0:
                    cand_state = "Unknown_Grade_Cancer_Proximity"
                else:
                    cand_state = "No_Recorded_Core_Correspondence"

                all_candidates.append({
                    "Candidate_UID": f"{model_name}_CustomDyn_{sid}_cand{rank}",
                    "Model": model_name,
                    "Extractor": "Custom_Dynamic_Cube",
                    "Session_ID": sid,
                    "Patient_ID": pid,
                    "Is_Canonical": is_canon,
                    "Candidate_Rank": rank,
                    "Max_Prob": cb["p_max"],
                    "Mean_Prob": float(np.mean(prob_map[cb["mask_c"]])),
                    "Volume_cc": cb["vol_cc"],
                    "Voxel_Count": cb["n_vox"],
                    "CoM_LPS_X_mm": round(com_lps[0], 3),
                    "CoM_LPS_Y_mm": round(com_lps[1], 3),
                    "CoM_LPS_Z_mm": round(com_lps[2], 3),
                    "Peak_LPS_X_mm": round(peak_lps[0], 3),
                    "Peak_LPS_Y_mm": round(peak_lps[1], 3),
                    "Peak_LPS_Z_mm": round(peak_lps[2], 3),
                    "CoM_Peak_Offset_mm": round(offset_mm, 3),
                    "CoM_Dist_csPCa_mm": round(com_d_cs, 3) if np.isfinite(com_d_cs) else np.nan,
                    "Peak_Dist_csPCa_mm": round(peak_d_cs, 3) if np.isfinite(peak_d_cs) else np.nan,
                    "Surf3D_Dist_csPCa_mm": round(surf_d_cs, 3) if np.isfinite(surf_d_cs) else np.nan,
                    "CoM_Dist_GG1_mm": round(com_d_gg1, 3) if np.isfinite(com_d_gg1) else np.nan,
                    "CoM_Dist_Benign_mm": round(com_d_ben, 3) if np.isfinite(com_d_ben) else np.nan,
                    "Candidate_State": cand_state,
                })


        if (idx + 1) % 25 == 0 or idx == n_total - 1:
            print(f"Processed {idx+1}/{n_total} sessions ({(time.time()-t0):.1f}s, {len(all_candidates)} candidates extracted)...", flush=True)

    df_cand_master = pd.DataFrame(all_candidates)
    cands_csv = os.path.join(AUDIT_DIR, "CANDIDATES_ANALYSIS_READY.csv")
    df_cand_master.to_csv(cands_csv, index=False)
    print(f"\nSaved CANDIDATES_ANALYSIS_READY.csv ({len(df_cand_master)} rows) -> {cands_csv}")

    # Summary table
    print("\nSummary of Extracted Candidates:")
    print(df_cand_master.groupby(["Model", "Extractor"])["Candidate_Rank"].count())

if __name__ == "__main__":
    run_unified_candidate_extraction()
