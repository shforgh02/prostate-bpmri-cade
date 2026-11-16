# Prostate MRI deep learning candidates and tracked biopsy cores

Code for a retrospective study that asks what prostate MRI deep learning models add to the biopsy targets that radiologists have already drawn. Model output is compared with the recorded positions and histologic results of MRI–US fusion biopsy cores, at radiologist targets and away from them.

## What the code does

1. Trains a 3D SegResNet on the public PI-CAI training set and tests it on a held-out fold.
2. Prepares the public UCLA Prostate-MRI-US-Biopsy collection: image standardization, biopsy core coordinates, radiologist targets, clinical variables.
3. Applies three models to the same T2-weighted, ADC, and high-b-value images without retraining:
   - SegResNet (trained here),
   - the public PI-CAI nnU-Net baseline,
   - ProViCNet (published weights).
4. Converts each probability map to ranked lesion candidates with report-guided annotation.
5. Evaluates the candidates against tracked biopsy cores:
   - localization of cancers, compared with a chance model;
   - association between model–target agreement and cancer at the radiologist target;
   - cancer in systematic cores away from radiologist targets, compared with a chance model;
   - examination-level and target-level discrimination in addition to suspicion score and PSA density.
6. Repeats the agreement analysis for ProViCNet in the PROSTATEx training set (findings sampled by MR-guided biopsy), using the PI-CAI copies of the same examinations.

The chance model exchanges candidate sets between patients with the same number of candidates and maps them through the prostate bounding box. Cores and targets from one patient are never treated as independent (permutation between patients, patient bootstrap, or generalized estimating equations clustered by patient).

## Data

No data are stored in this repository. All data are public and must be downloaded by the user.

| Data | Source | Used for |
|---|---|---|
| Prostate-MRI-US-Biopsy (UCLA) | The Cancer Imaging Archive, doi:10.7937/TCIA.2020.A61IOC1A | Study sample: MRI, prostate and target surfaces, tracked cores, histology |
| PI-CAI public training set | https://pi-cai.grand-challenge.org, doi:10.5281/zenodo.6624726 | Training and internal test of SegResNet |
| PROSTATEx lesion tables (training set) | The Cancer Imaging Archive, https://www.cancerimagingarchive.net/collection/prostatex/ | External replication: finding positions, zones, labels |

Set the environment variable `PROSTATE_DATA_ROOT` to the folder that holds the data. The scripts expect this layout:

```
$PROSTATE_DATA_ROOT/
├── MRI_US_Biopsy/
│   ├── manifest-1694710246744/                 DICOM series from TCIA
│   ├── PKG - Prostate-MRI-US-Biopsy-STL/       prostate and target surfaces
│   ├── TCIA-Biopsy-Data_2020-07-14.xlsx        core coordinates and histology
│   └── Target-Data_2019-12-05-2.xlsx           radiologist targets
└── PI-CAI_Validation/
    ├── picai_public_images_fold0 ... fold4/
    └── picai_labels-main/
```

Please follow the license and citation terms of each data set.

## Models

| Model | Origin | Notes |
|---|---|---|
| SegResNet | Trained with `training/train_segresnet_modal.py` | Weights are not included in this repository |
| nnU-Net | Public PI-CAI baseline (container `joeranbosma/picai_nnunet`) | Run with the PI-CAI preprocessing and test-time augmentation |
| ProViCNet | https://github.com/pimed/ProViCNet (commit `1b016ac`), weights on Hugging Face | Fusion model, clinically significant cancer channel |

## Repository structure

```
├── training/                  SegResNet training and internal test (Modal, GPU)
├── pipeline/                  UCLA preprocessing, SegResNet inference, clinical reference tables, candidates
│   ├── 01_phase1_preprocessing_and_standardization.py
│   ├── 02_phase2_model_inference_and_internal_eval.py
│   ├── 03_build_clinical_reference_and_manifest.py
│   ├── 04_extract_lesion_candidates_unified.py
│   └── 05_run_burden_matched_and_statistical_analysis.py
├── docs/                      excluded examinations, target-to-score table
├── analysis/
│   ├── _common.py             shared loaders and geometry helpers
│   ├── H1, H2                 DICOM header audits (b values, scanner, field strength)
│   ├── B0, B7, B7b            technical exclusions (header rule, image-content rule)
│   ├── B1–B5                  inputs for nnU-Net and ProViCNet, inference on Modal, candidates
│   ├── A1–A3                  localization vs chance, cores away from targets, examination level
│   ├── B6c                    links every radiologist target surface to its own suspicion score
│   ├── B6, B6b                agreement between models and radiologist targets
│   ├── H3                     biopsy sessions per examination (single-session sensitivity analysis)
│   ├── F1                     later biopsy sessions in men without cancer at the first session
│   ├── G1                     localization at a matched number of candidates; sampling of candidates by cores
│   ├── B8, C2                 classifier trained on core histology (frozen ProViCNet features)
│   ├── C1, C3–C5, E1          summary table, multiplicity adjustment, further analyses
│   ├── R1, R1b                PROSTATEx patients matched to their PI-CAI copies (R1b: with the half-voxel origin conversion)
│   ├── R2–R5                  replication: inputs, ProViCNet on Modal, candidates, agreement analysis
│   ├── D1–D3, D7, D9          figures and tables
│   └── extended/              further analyses (X0–X9)
├── requirements.txt           minimum versions
├── requirements-lock.txt      exact versions used for the analyses (Python 3.12.9)
├── CITATION.cff
└── LICENSE                    MIT
```

## Setup

```bash
conda create -n prostate_dl python=3.12 -y
conda activate prostate_dl
pip install -r requirements.txt          # or requirements-lock.txt for the exact versions
export PROSTATE_DATA_ROOT="/path/to/data"      # PowerShell: $env:PROSTATE_DATA_ROOT = "D:\data"
mkdir -p outputs figures tables
```

Run every script from the repository root. Generated files go to `results/`, `sources/`, `outputs/`, `figures/`, and `tables/`, which are ignored by git.

## Workflow

**1. SegResNet (optional if weights are available)**

```bash
modal run training/train_segresnet_modal.py
modal run training/evaluate_segresnet_internal_modal.py
```

**2. Study sample and SegResNet candidates**

```bash
python pipeline/01_phase1_preprocessing_and_standardization.py
python pipeline/02_phase2_model_inference_and_internal_eval.py
python pipeline/03_build_clinical_reference_and_manifest.py
python pipeline/04_extract_lesion_candidates_unified.py
python pipeline/05_run_burden_matched_and_statistical_analysis.py
```

**3. Technical exclusions**

```bash
python analysis/H1_highb_series_audit.py "$PROSTATE_DATA_ROOT/MRI_US_Biopsy" outputs/highb_series_audit.csv
python analysis/H2_scanner_coil_headers.py "$PROSTATE_DATA_ROOT/MRI_US_Biopsy" outputs/scanner_coil_by_session.csv
python analysis/B0_exclusions.py            # header rule  -> outputs/EXCLUDED_SESSIONS.csv
python analysis/B1_build_native_inputs.py   # inputs at PI-CAI resolution
python analysis/B1b_build_gland_masks.py
python analysis/B7_input_content_qc.py      # content rule -> outputs/B7_input_content_qc.csv
python analysis/B7b_combine_exclusions.py   # both rules  -> outputs/EXCLUDED_SESSIONS_v2.csv
```

**4. nnU-Net and ProViCNet (GPU; written for Modal)**

```bash
python analysis/_modal_sync.py up <volume> <local inputs folder> <remote folder>
python analysis/B2_modal_nnunet_official.py timing
python analysis/B2_modal_nnunet_official.py full 4
modal run --detach analysis/B4_modal_provicnet.py --mode run
python analysis/B3_probmaps_to_session_grid.py --kind nnunet --pred <folder> --inputs <imagesTs folder> --out results/nnunet_official_probmaps
python analysis/B5_extract_candidates_generic.py --model nnUNet_official --dir results/nnunet_official_probmaps --out <csv>
```

Repeat B3 and B5 for ProViCNet, then concatenate the candidate tables of the three models into `outputs/CANDIDATES_3MODELS.csv`. The Modal scripts read the workspace from `MODAL_PROFILE`; volume names are set at the top of each script.

**5. Main analyses**

```bash
ARGS="--exclude outputs/EXCLUDED_SESSIONS_v2.csv --candidates outputs/CANDIDATES_3MODELS.csv --tag final_qc"
python analysis/A1_chance_models_and_tolerance.py $ARGS
python analysis/A2_core_level_off_target.py $ARGS --nperm 2000
python analysis/A3_agreement_and_psa_density.py $ARGS
python analysis/B6c_link_target_scores.py     # once; -> outputs/TARGET_SCORES.csv
python analysis/B6_per_target_agreement.py $ARGS
python analysis/C1_summary_table.py --tag final_qc
python analysis/C4_holm.py
python analysis/C5_target_level_discrimination.py --tag final_qc
```

The same commands with `--exclude outputs/EXCLUDED_SESSIONS.csv --tag final` give the sensitivity analysis before the image-content exclusion.

```bash
python analysis/H3_biopsy_sessions.py        # session dates; writes the exclusion list for the single-session analysis
python analysis/F1_followup_biopsy.py        # later biopsy sessions
python analysis/G1_burden_and_sampling.py    # matched number of candidates; sampling of candidates by cores
```

**6. External replication (PROSTATEx, ProViCNet)**

Download `ProstateX-TrainingLesionInformationv2.zip` from the PROSTATEx collection page and unpack it to `outputs/replication/raw/`.

```bash
python analysis/R1_match_prostatex_to_picai.py      # writes the PI-CAI T2 header table (its own match fails, see R1b)
python analysis/R1b_match_prostatex_to_picai.py     # match with origin converted to the voxel center; matched patients and coordinate check
python analysis/R2_build_prostatex_inputs.py        # inputs, gland masks, 1-mm grids, image-content rule
python analysis/_modal_sync.py up prostatex-provicnet-vol <PROSTATEx_replication/imagesTs> inputs
python analysis/_modal_sync.py up prostatex-provicnet-vol <PROSTATEx_replication/masks> inputs
modal run analysis/R3_modal_provicnet_prostatex.py --mode run
python analysis/_modal_sync.py down prostatex-provicnet-vol pred <PROSTATEx_replication/pred> .nii.gz
python analysis/R4_prostatex_candidates.py
python analysis/R5_prostatex_agreement.py           # agreement analysis in PROSTATEx
```

`PROSTATEx_replication/` is created next to the data folders under `PROSTATE_DATA_ROOT`.

**7. Further analyses, figures, and tables**

The header of each remaining script (`B8`, `C2`, `C3`, `E1`, `R1`, `extended/X0`–`X9`, `D1`–`D9`) states its inputs and the command to run it. `D2_flow_figure.py` reads `outputs/FLOW_NUMBERS.txt`, a short text file of cohort counts that is assembled from the logs of the steps above.

## Notes

- Coordinates: biopsy cores and candidate centers are handled in LPS millimeters. SimpleITK uses LPS and nibabel uses RAS; the scripts convert explicitly.
- Diffusion series are checked by the b value in the DICOM header, not by the series name.
- Suspicion scores: the public target table lists a score for every numbered target. `B6c` links each target surface to its mask region; the examination-level score is the highest target score.
- The PROSTATEx images table stores the image origin at the corner of the first voxel; image files store it at the voxel center (half a voxel apart).
- The GPU steps were run on Modal with T4 and A10G instances. They can be adapted to any GPU machine by calling the same functions locally.
- Paths in file headers use forward slashes; the scripts were developed on Windows and also run under Git Bash.

## Citation and license

The code is released under the MIT License (`LICENSE`). To cite it, use `CITATION.cff` ("Cite this repository" on GitHub) or the Zenodo record of release v1.0.0. A citation of the article will be added after publication.

Third-party components keep their own licenses: MONAI, nnU-Net, the PI-CAI baseline and `report-guided-annotation`, and ProViCNet.
