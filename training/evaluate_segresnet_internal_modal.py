import modal

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "monai", "pandas", "numpy", "SimpleITK", "scikit-learn", "matplotlib", "seaborn", "tqdm", "scipy", "nibabel")
)

app = modal.App("picai-unified-evaluation-new")
volume = modal.Volume.from_name("picai-dataset-vol")

@app.function(
    image=image, 
    gpu="A10G",           
    timeout=14400, # 4 hours
    memory=32768,
    volumes={"/data": volume} 
)
def run_unified_evaluation_pipeline():
    import os
    import time
    import torch
    import numpy as np
    import pandas as pd
    import SimpleITK as sitk
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.ndimage import label, center_of_mass, binary_dilation
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
    from sklearn.calibration import calibration_curve
    import nibabel as nib
    from tqdm import tqdm
    import warnings
    
    warnings.filterwarnings('ignore')
    
    print(f"\n{'='*90}")
    print(f" 🎯 SOTA UNIFIED 3-CHANNEL EVALUATION, OPTIMIZATION & STORYBOARD PIPELINE")
    print(f"{'='*90}")

    # =====================================================================
    # SOTA CONFIGURATION: CLINICAL STANDARD OVERRIDE
    # =====================================================================
    USE_CLINICAL_STATIC_THRESHOLD = True
    STATIC_THRESHOLD = 0.10 # Strictly matches your manuscript's highly sensitive clinical standard

    # ---------------------------------------------------------
    # CLOUD PATHS
    # ---------------------------------------------------------
    DATA_DIR = "/data/standardized_picai_sessions_new"
    MASTER_CSV = "/data/picai_master_clinical_cohort_new.csv"
    MODEL_PATH = "/data/models/picai_segresnet_best_loss_new.pth"
    
    OUT_CSV = "/data/Phase2_Blob_Candidates_Isotropic_New.csv"
    OUTPUT_DIR = "/data/Publication_Figures_PICAI_New"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    PATCH_SIZE = (32, 64, 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================================
    # STEP 1: CANDIDATE EXTRACTION (3-CHANNEL SEGRESNET)
    # =====================================================================
    print("\n[Step 1/5] Extracting 3-Channel Candidates from Validation & Testing Splits...")
    
    from monai.networks.nets import SegResNet
    from monai.inferers import sliding_window_inference
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd, CropForegroundd, EnsureTyped

    model = SegResNet(
        spatial_dims=3, in_channels=3, out_channels=1, 
        init_filters=32, dropout_prob=0.2, norm="instance"
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    df_master = pd.read_csv(MASTER_CSV)
    eval_df = df_master[df_master['Fold'].isin(['picai_public_images_fold3', 'picai_public_images_fold4'])].copy()

    inference_transforms = Compose([
        LoadImaged(keys=["t2", "adc", "dwi", "pros", "label"]),
        EnsureChannelFirstd(keys=["t2", "adc", "dwi", "pros", "label"]),
        ConcatItemsd(keys=["t2", "adc", "dwi"], name="image"),
        CropForegroundd(keys=["image", "label", "pros"], source_key="pros", margin=(15, 15, 15)),
        EnsureTyped(keys=["image", "label", "pros"], data_type="tensor")
    ])

    blob_dataset = []
    t_start = time.time()

    if not os.path.exists(OUT_CSV):
        with torch.no_grad():
            for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Processing sessions"):
                sess_id = row['Session_ID']
                pid = row['Patient_ID']
                fold = row['Fold']
                log_psad = row['Log_PSAD']
                sdir = os.path.join(DATA_DIR, sess_id)

                data_dict = {
                    "t2": os.path.join(sdir, "t2_norm.nii.gz"),
                    "adc": os.path.join(sdir, "adc_norm.nii.gz"),
                    "dwi": os.path.join(sdir, "dwi_norm.nii.gz"),
                    "pros": os.path.join(sdir, "prostate_mask.nii.gz"),
                    "label": os.path.join(sdir, "target_1_mask.nii.gz")
                }

                if not os.path.exists(data_dict["t2"]): continue

                transformed = inference_transforms(data_dict)
                image_tensor = transformed["image"].unsqueeze(0).to(device)
                gt_arr = transformed["label"].squeeze().cpu().numpy() > 0
                pros_arr = transformed["pros"].squeeze().cpu().numpy() > 0
                
                t2_arr = transformed["image"][0].cpu().numpy()
                adc_arr = transformed["image"][1].cpu().numpy()
                dwi_arr = transformed["image"][2].cpu().numpy()

                with torch.cuda.amp.autocast():
                    logits = sliding_window_inference(image_tensor, PATCH_SIZE, 4, model, overlap=0.5, mode="gaussian")
                
                heatmap = torch.sigmoid(logits).squeeze().cpu().numpy()

                dilated_pros = binary_dilation(pros_arr, iterations=4)
                heatmap[~dilated_pros] = 0.0

                labeled_blobs, num_features = label(heatmap >= 0.05)
                
                for feat_idx in range(1, num_features + 1):
                    blob_mask = (labeled_blobs == feat_idx)
                    vol_cc = blob_mask.sum() * 0.001
                    
                    if vol_cc < 0.03 or vol_cc > 25.0: continue
                    
                    b_prob = heatmap[blob_mask]
                    cent_vox = center_of_mass(blob_mask)

                    adc_vals = adc_arr[blob_mask]
                    t2_vals = t2_arr[blob_mask]
                    dwi_vals = dwi_arr[blob_mask]

                    is_hit = 0
                    dice_score = 0.0
                    if gt_arr.sum() > 0:
                        intersection = np.logical_and(blob_mask, gt_arr).sum()
                        if intersection > 0:
                            gt_vol_cc = gt_arr.sum() * 0.001
                            if (intersection * 0.001) / gt_vol_cc >= 0.10: 
                                is_hit = 1
                            dice_score = (2.0 * intersection) / (blob_mask.sum() + gt_arr.sum())

                    blob_dataset.append({
                        'Session_ID': sess_id,
                        'Patient_ID': pid,
                        'Fold': fold,
                        'Blob_Vol_cc': round(vol_cc, 4),
                        'Z_Loc_Relative': round(cent_vox[0] / heatmap.shape[0], 4),
                        'Max_Prob': round(np.max(b_prob), 4),
                        'Mean_Prob': round(np.mean(b_prob), 4),
                        'Log_PSAD': round(log_psad, 4),
                        'ADC_Mean': round(np.mean(adc_vals), 4),
                        'ADC_Min': round(np.min(adc_vals), 4),
                        'DWI_Mean': round(np.mean(dwi_vals), 4),
                        'DWI_Max': round(np.max(dwi_vals), 4),
                        'Is_True_Hit': is_hit,
                        'Dice_Score': round(dice_score, 4)
                    })

        df_blobs = pd.DataFrame(blob_dataset)
        df_blobs.to_csv(OUT_CSV, index=False)
        print(f"✅ Extracted {len(df_blobs)} raw candidates. Saved to: {OUT_CSV}")
    else:
        df_blobs = pd.read_csv(OUT_CSV)
        print(f"✅ Loaded {len(df_blobs)} existing raw candidates from Volume.")

    # =====================================================================
    # STEP 2: COMPUTATIONAL NOISE FLOOR 
    # =====================================================================
    print("\n[Step 2/5] Applying Computational Noise Floor...")
    best_threshold = 0.10 
    print(f"   -> Noise Floor set at: {best_threshold:.2f}")

    # =====================================================================
    # STEP 3: STATISTICAL TEST METRICS (FOLD 4 ZERO-SHOT WITH 95% CIs)
    # =====================================================================
    print("\n[Step 3/5] Computing Sequestered Test Set (Fold 4) Metrics...")
    test_master = df_master[df_master['Fold'] == 'picai_public_images_fold4'].copy()
    test_blobs_raw = df_blobs[df_blobs['Fold'] == 'picai_public_images_fold4'].copy()
    
    test_blobs_thresholded = test_blobs_raw[test_blobs_raw['Max_Prob'] >= best_threshold].copy()

    test_total_patients = test_master['Patient_ID'].nunique()
    test_cancer_patients = test_master[test_master['Clinical_csPCa'] == 1]['Patient_ID'].nunique()

    # Continuous patient-level arrays for ROC/PR/Brier
    y_true_patient, y_score_patient = [], []
    for _, row in test_master.iterrows():
        pid = row['Patient_ID']
        has_cancer = int(row['Clinical_csPCa'])
        p_blobs = test_blobs_raw[test_blobs_raw['Patient_ID'] == pid]
        
        y_true_patient.append(has_cancer)
        y_score_patient.append(p_blobs['Max_Prob'].max() if not p_blobs.empty else 0.0)

    y_true_patient = np.array(y_true_patient)
    y_score_patient = np.array(y_score_patient)

    # Standalone continuous blob-level arrays
    y_true_blob = test_blobs_raw['Is_True_Hit'].values
    y_score_blob = test_blobs_raw['Max_Prob'].values

    # Statistical Bootstrapping Engine (1,000 Iterations for 95% CIs)
    def bootstrap_metric_ci(y_t, y_s, metric_func, n_bootstraps=1000):
        rng = np.random.RandomState(42)
        values = []
        for _ in range(n_bootstraps):
            indices = rng.choice(len(y_t), len(y_t), replace=True)
            if len(np.unique(y_t[indices])) < 2: continue
            values.append(metric_func(y_t[indices], y_s[indices]))
        return np.percentile(values, [2.5, 97.5])

    def calc_auc(y_t, y_s):
        fpr, tpr, _ = roc_curve(y_t, y_s)
        return auc(fpr, tpr)

    # 1. Patient-level statistics
    auc_pat = calc_auc(y_true_patient, y_score_patient)
    auc_pat_ci = bootstrap_metric_ci(y_true_patient, y_score_patient, calc_auc)
    
    pr_pat = average_precision_score(y_true_patient, y_score_patient)
    pr_pat_ci = bootstrap_metric_ci(y_true_patient, y_score_patient, average_precision_score)
    brier_pat = brier_score_loss(y_true_patient, y_score_patient)

    # 2. Blob-level statistics
    auc_blob = calc_auc(y_true_blob, y_score_blob)
    auc_blob_ci = bootstrap_metric_ci(y_true_blob, y_score_blob, calc_auc)
    
    pr_blob = average_precision_score(y_true_blob, y_score_blob)
    pr_blob_ci = bootstrap_metric_ci(y_true_blob, y_score_blob, average_precision_score)
    brier_blob = brier_score_loss(y_true_blob, y_score_blob)

    # 3. Top-K binary retrieval statistics
    def bootstrap_top_k_ci(k_val, n_bootstraps=1000):
        rng = np.random.RandomState(42)
        patient_ids = test_master['Patient_ID'].unique()
        sensitivities = []
        fps_rates = []
        
        for _ in range(n_bootstraps):
            sample_pids = rng.choice(patient_ids, len(patient_ids), replace=True)
            
            sample_hits = 0
            sample_cancers = 0
            sample_fps = 0
            
            for pid in sample_pids:
                is_cancer = test_master[test_master['Patient_ID'] == pid]['Clinical_csPCa'].iloc[0] == 1
                if is_cancer: sample_cancers += 1
                
                p_blobs = test_blobs_thresholded[test_blobs_thresholded['Patient_ID'] == pid].sort_values('Max_Prob', ascending=False).head(k_val)
                if is_cancer and not p_blobs.empty and p_blobs['Is_True_Hit'].sum() > 0:
                    sample_hits += 1
                    
                sample_fps += len(p_blobs) - p_blobs['Is_True_Hit'].sum()
                
            sensitivities.append((sample_hits / sample_cancers) * 100 if sample_cancers > 0 else 0)
            fps_rates.append(sample_fps / len(sample_pids))
            
        return np.percentile(sensitivities, [2.5, 97.5]), np.percentile(fps_rates, [2.5, 97.5])

    print("\n" + "-"*80)
    print(" 📋 SOTA TEST SET REPORT FOR MANUSCRIPT (FOLD 4 ZERO-SHOT)")
    print("-"*80)
    true_hit_blobs = test_blobs_thresholded[test_blobs_thresholded['Is_True_Hit'] == 1]
    mean_dice = true_hit_blobs['Dice_Score'].mean()
    std_dice = true_hit_blobs['Dice_Score'].std()
    print(f"Voxel-Level Dice (Hits Only)  : {mean_dice:.3f} ± {std_dice:.3f}")
    
    print(f"\nPATIENT-LEVEL DISCRIMINATION (N={test_total_patients} patients):")
    print(f"  -> Patient ROC-AUC          : {auc_pat:.3f} (95% CI: {auc_pat_ci[0]:.3f} - {auc_pat_ci[1]:.3f})")
    print(f"  -> Patient PR-AUC           : {pr_pat:.3f} (95% CI: {pr_pat_ci[0]:.3f} - {pr_pat_ci[1]:.3f})")
    print(f"  -> Brier Score              : {brier_pat:.3f}")
    
    print(f"\nBLOB-LEVEL (TARGET) DISCRIMINATION (N={len(test_blobs_raw)} extracted blobs):")
    print(f"  -> Blob ROC-AUC             : {auc_blob:.3f} (95% CI: {auc_blob_ci[0]:.3f} - {auc_blob_ci[1]:.3f})")
    print(f"  -> Blob PR-AUC              : {pr_blob:.3f} (95% CI: {pr_blob_ci[0]:.3f} - {pr_blob_ci[1]:.3f})")
    print(f"  -> Blob Brier Score         : {brier_blob:.3f}")

    print(f"\nCLINICAL TOP-K RETRIEVAL PERFORMANCE (Active Threshold = {best_threshold:.2f}):")
    for k in [1, 2, 3]:
        patient_hits = 0
        total_fps = 0
        for pid in test_master['Patient_ID'].unique():
            is_cancer = test_master[test_master['Patient_ID'] == pid]['Clinical_csPCa'].iloc[0] == 1
            p_blobs = test_blobs_thresholded[test_blobs_thresholded['Patient_ID'] == pid].sort_values('Max_Prob', ascending=False).head(k)
            if is_cancer and not p_blobs.empty and p_blobs['Is_True_Hit'].sum() > 0:
                patient_hits += 1
            total_fps += len(p_blobs) - p_blobs['Is_True_Hit'].sum()
            
        sens = (patient_hits / test_cancer_patients) * 100
        fps = total_fps / test_total_patients
        sens_ci, fps_ci = bootstrap_top_k_ci(k)
        
        print(f"  Top {k} Target Boundary Output:")
        print(f"    -> Sensitivity            : {sens:.1f}% (95% CI: {sens_ci[0]:.1f}% - {sens_ci[1]:.1f}%)")
        print(f"    -> FP / Scan Rate         : {fps:.2f} (95% CI: {fps_ci[0]:.2f} - {fps_ci[1]:.2f})")

    # =====================================================================
    # STEP 4: PLOT SCIENTIFIC CHARTS (CONTINUOUS SPECTRUM)
    # =====================================================================
    print("\n[Step 4/5] Generating Scientific Charts...")
    
    # --- CHART 1: PATIENT LEVEL ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    fpr, tpr, _ = roc_curve(y_true_patient, y_score_patient)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'Patient ROC (AUC = {auc_pat:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_title('A. Receiver Operating Characteristic', fontweight='bold')
    axes[0].legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_true_patient, y_score_patient)
    axes[1].plot(recall, precision, color='purple', lw=2, label=f'Patient PR (AUC = {pr_pat:.3f})')
    axes[1].axhline(y_true_patient.sum()/len(y_true_patient), color='gray', linestyle='--')
    axes[1].set_title('B. Precision-Recall Curve', fontweight='bold')
    axes[1].legend(loc="upper right")

    prob_true, prob_pred = calibration_curve(y_true_patient, y_score_patient, n_bins=10)
    axes[2].plot(prob_pred, prob_true, marker='o', color='teal', lw=2, label=f'Brier = {brier_pat:.3f}')
    axes[2].plot([0, 1], [0, 1], linestyle='--', color='black')
    axes[2].set_title('C. Patient-Level Calibration', fontweight='bold')
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig1_SOTA_Test_Set_Stats_Patient.png"))
    plt.close()

    # --- CHART 2: BLOB (TARGET) LEVEL ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    fpr_b, tpr_b, _ = roc_curve(y_true_blob, y_score_blob)
    axes2[0].plot(fpr_b, tpr_b, color='firebrick', lw=2, label=f'Blob ROC (AUC = {auc_blob:.3f})')
    axes2[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes2[0].set_title('A. Target-Level Receiver Operating Characteristic', fontweight='bold')
    axes2[0].legend(loc="lower right")

    precision_b, recall_b, _ = precision_recall_curve(y_true_blob, y_score_blob)
    axes2[1].plot(recall_b, precision_b, color='darkgreen', lw=2, label=f'Blob PR (AUC = {pr_blob:.3f})')
    axes2[1].axhline(y_true_blob.sum()/len(y_true_blob), color='gray', linestyle='--')
    axes2[1].set_title('B. Target-Level Precision-Recall Curve', fontweight='bold')
    axes2[1].legend(loc="upper right")

    prob_true_b, prob_pred_b = calibration_curve(y_true_blob, y_score_blob, n_bins=10)
    axes2[2].plot(prob_pred_b, prob_true_b, marker='o', color='royalblue', lw=2, label=f'Brier = {brier_blob:.3f}')
    axes2[2].plot([0, 1], [0, 1], linestyle='--', color='black')
    axes2[2].set_title('C. Target-Level Calibration', fontweight='bold')
    axes2[2].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_SOTA_Test_Set_Stats_Blob.png"))
    plt.close()

    # =====================================================================
    # STEP 5: EXTRACTION AND 600 DPI STORYBOARD GENERATION
    # =====================================================================
    print("\n[Step 5/5] Extracting Extreme Cases and Generating 600 DPI Storyboards...")
    
    hits_test = test_blobs_thresholded[test_blobs_thresholded['Is_True_Hit'] == 1].copy()
    
    best_5 = hits_test.sort_values('Dice_Score', ascending=False).head(5)['Session_ID'].tolist()
    
    missed_pids = []
    for pid in test_master[test_master['Clinical_csPCa'] == 1]['Patient_ID'].unique():
        p_blobs = test_blobs_thresholded[test_blobs_thresholded['Patient_ID'] == pid]
        if p_blobs.empty or p_blobs['Is_True_Hit'].max() == 0:
            missed_pids.append(pid)
    worst_fn_5 = test_master[test_master['Patient_ID'].isin(missed_pids)].head(5)['Session_ID'].tolist()
    
    poor_hit_5 = hits_test[hits_test['Dice_Score'] > 0.10].sort_values('Dice_Score', ascending=True).head(5)['Session_ID'].tolist()
    
    fp_test = pd.merge(test_blobs_thresholded, test_master[['Session_ID', 'Clinical_csPCa']], on='Session_ID')
    worst_fp_5 = fp_test[fp_test['Clinical_csPCa'] == 0].sort_values('Max_Prob', ascending=False).head(5)['Session_ID'].tolist()

    storyboard_cohorts = [
        (best_5, "Concordant_Best_Dice"),
        (worst_fn_5, "False_Negative_Misses"),
        (poor_hit_5, "Suboptimal_Low_Dice_Hits"),
        (worst_fp_5, "False_Positive_BPH_Mimics")
    ]

    # Clinical Multisequence Visualization Engine (T2W + ADC + DWI)
    def generate_clinical_storyboard(session_id, cohort_folder):
        sess_dir = os.path.join(DATA_DIR, session_id)
        target_folder = os.path.join(OUTPUT_DIR, cohort_folder)
        os.makedirs(target_folder, exist_ok=True)
        
        t2_path = os.path.join(sess_dir, "t2_norm.nii.gz")
        adc_path = os.path.join(sess_dir, "adc_norm.nii.gz")
        dwi_path = os.path.join(sess_dir, "dwi_norm.nii.gz")
        pros_path = os.path.join(sess_dir, "prostate_mask.nii.gz")
        rad_path = os.path.join(sess_dir, "target_1_mask.nii.gz")

        if not os.path.exists(t2_path): return

        t2_sitk = sitk.ReadImage(t2_path)
        adc_sitk = sitk.ReadImage(adc_path)
        dwi_sitk = sitk.ReadImage(dwi_path)
        pros_sitk = sitk.ReadImage(pros_path)
        
        t2_arr = sitk.GetArrayFromImage(t2_sitk)
        adc_arr = sitk.GetArrayFromImage(adc_sitk)
        dwi_arr = sitk.GetArrayFromImage(dwi_sitk)
        
        raw_pros_arr = sitk.GetArrayFromImage(pros_sitk)

        data_dict = {"t2": t2_path, "adc": adc_path, "dwi": dwi_path, "pros": pros_path, "label": rad_path}
        transformed = inference_transforms(data_dict)
        image_tensor = transformed["image"].unsqueeze(0).to(device)
        
        cropped_pros_arr = transformed["pros"].squeeze().cpu().numpy() > 0
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = sliding_window_inference(image_tensor, PATCH_SIZE, 4, model, overlap=0.5, mode="gaussian")
        
        heatmap = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        heatmap[~binary_dilation(cropped_pros_arr, iterations=4)] = 0.0
        ai_mask = (heatmap >= best_threshold).astype(np.float32)

        affine = transformed["image"].meta.get("affine", np.eye(4))
        affine = affine.cpu().numpy() if torch.is_tensor(affine) else affine
        temp_path = os.path.join(OUTPUT_DIR, f"temp_{session_id}.nii.gz")
        nib.save(nib.Nifti1Image(ai_mask, affine=affine), temp_path)
        
        ai_sitk = sitk.ReadImage(temp_path)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(t2_sitk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        ai_resampled = resampler.Execute(ai_sitk)
        ai_arr = sitk.GetArrayFromImage(ai_resampled)
        
        has_rad = False
        if os.path.exists(rad_path):
            rad_sitk = sitk.ReadImage(rad_path)
            rad_arr = sitk.GetArrayFromImage(rad_sitk)
            if rad_arr.sum() > 0: has_rad = True

        ai_z = np.sum(ai_arr, axis=(1,2))
        rad_z = np.sum(rad_arr, axis=(1,2)) if has_rad else np.zeros_like(ai_z)
        both_exist = (ai_z > 0) & (rad_z > 0)
        
        if np.any(both_exist):
            z_slice = np.argmax((ai_z + rad_z) * both_exist)
        elif np.sum(ai_z) > 0:
            z_slice = np.argmax(ai_z)
        elif has_rad and np.sum(rad_z) > 0:
            z_slice = np.argmax(rad_z)
        else:
            z_slice = np.argmax(np.sum(raw_pros_arr, axis=(1,2)))
            
        z_slice = int(np.clip(z_slice, 0, t2_arr.shape[0]-1))

        y_idx, x_idx = np.where(raw_pros_arr[z_slice] > 0)
        if len(y_idx) > 0 and len(x_idx) > 0:
            pad = 45
            ymin, ymax = max(0, np.min(y_idx) - pad), min(t2_arr.shape[1], np.max(y_idx) + pad)
            xmin, xmax = max(0, np.min(x_idx) - pad), min(t2_arr.shape[2], np.max(x_idx) + pad)
        else:
            ymin, ymax, xmin, xmax = 0, t2_arr.shape[1], 0, t2_arr.shape[2]

        def apply_window(img_2d):
            p1, p99 = np.percentile(img_2d, (1, 99))
            img_clipped = np.clip(img_2d, p1, p99)
            return (img_clipped - p1) / (p99 - p1 + 1e-6)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=600)
        fig.patch.set_facecolor('black')
        
        COLOR_AI = '#FF8C00'      
        COLOR_HUMAN = '#00BFFF'   

        titles = ["A. T2-Weighted (Anatomy)", "B. ADC Map (Water Restriction)", "C. DWI (High b-value)"]
        arrays = [t2_arr, adc_arr, dwi_arr]
        cmaps = ['gray', 'bone', 'viridis']

        for ax, img_arr, title, cmap in zip(axes, arrays, titles, cmaps):
            img_2d = apply_window(img_arr[z_slice, :, :])
            ax.imshow(img_2d, cmap=cmap, interpolation='bicubic')
            ax.set_title(title, fontsize=14, color='white', fontweight='bold')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)
            ax.axis('off')

            if has_rad and np.sum(rad_arr[z_slice, :, :]) > 0:
                ax.contour(rad_arr[z_slice, :, :], levels=[0.5], colors=[COLOR_HUMAN], linewidths=2.5)
                ax.plot([], [], color=COLOR_HUMAN, linewidth=2.5, label='Expert Target')

            if np.sum(ai_arr[z_slice, :, :]) > 0:
                ax.contour(ai_arr[z_slice, :, :], levels=[0.5], colors=[COLOR_AI], linewidths=2.5)
                ax.contourf(ai_arr[z_slice, :, :], levels=[0.5, 2.0], colors=[COLOR_AI], alpha=0.25)
                ax.plot([], [], color=COLOR_AI, linewidth=2.5, label='SOTA CADe Target')

        handles, labels = axes[2].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            axes[2].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, 
                           framealpha=0.8, facecolor='black', edgecolor='white', labelcolor='white')

        plt.suptitle(f"Case: {session_id} | Cohort: {cohort_folder.replace('_', ' ')}", 
                     fontsize=16, fontweight='bold', color='white', y=0.98)
        
        plt.tight_layout()
        save_path = os.path.join(target_folder, f"{session_id}.png")
        plt.savefig(save_path, bbox_inches='tight', facecolor='black', dpi=600)
        plt.close()

        if os.path.exists(temp_path): os.remove(temp_path)

    for candidate_list, folder in storyboard_cohorts:
        print(f"  -> Plotting {len(candidate_list)} storyboards for category: {folder}")
        for sid in candidate_list:
            generate_clinical_storyboard(sid, folder)

    print(f"\n" + "="*90)
    print(f" ✅ ALL PUBLICATION FIGURES & COHORTS GENERATED IN VOLUME: {OUTPUT_DIR}")
    print(f"{'='*90}")

@app.local_entrypoint()
def run():
    run_unified_evaluation_pipeline.remote()