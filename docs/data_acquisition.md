# Data Acquisition Guide

This project uses **only two colorectal cancer CT datasets**:

1. Medical Segmentation Decathlon (MSD) — Task10 Colon
2. StageII-Colorectal-CT — The Cancer Imaging Archive (TCIA)

The purpose of this document is to provide a reproducible acquisition and organization workflow for both datasets.

---

# 1. Medical Segmentation Decathlon — Task10 Colon

## 1.1 What the dataset contains

MSD Task10 Colon is a portal-venous phase abdominal CT dataset for **primary colon cancer** segmentation.

Key properties relevant to this project:

- 190 CT volumes in the challenge task;
- 126 public labeled training cases;
- 64 challenge test cases without public tumor labels;
- 3D CT volumes;
- expert primary-tumor segmentation for the public labeled cases;
- commonly distributed in NIfTI (`.nii.gz`) format.

For this project, the tumor segmentation is used **only for evaluation**. It must never be passed to the multimodal language model or used for training/tuning.

Official references:

- Medical Segmentation Decathlon: https://medicaldecathlon.com/
- AWS download page: https://medicaldecathlon.com/dataaws/
- MSD Nature Communications paper: https://doi.org/10.1038/s41467-022-30695-9

---

## 1.2 Direct download

A widely used official MSD archive for MONAI is:

```text
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar
```

Linux/macOS example:

```bash
mkdir -p data/raw/MSD_Task10_Colon
cd data/raw/MSD_Task10_Colon

wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar

tar -xf Task10_Colon.tar
```

Alternatively with `curl`:

```bash
curl -L \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar \
  -o Task10_Colon.tar

tar -xf Task10_Colon.tar
```

The official MSD AWS page can also be used if the direct mirror changes:

```text
https://medicaldecathlon.com/dataaws/
```

---

## 1.3 Expected directory structure

After extraction, the dataset should resemble:

```text
Task10_Colon/
├── dataset.json
├── imagesTr/
│   ├── colon_001.nii.gz
│   ├── colon_002.nii.gz
│   └── ...
├── labelsTr/
│   ├── colon_001.nii.gz
│   ├── colon_002.nii.gz
│   └── ...
└── imagesTs/
    ├── colon_XXX.nii.gz
    └── ...
```

Recommended project placement:

```text
data/raw/MSD_Task10_Colon/Task10_Colon/
```

Do not commit the raw dataset to GitHub.

---

## 1.4 Initial integrity checks

Recommended checks after download:

```bash
find data/raw/MSD_Task10_Colon/Task10_Colon/imagesTr -name '*.nii.gz' | wc -l
find data/raw/MSD_Task10_Colon/Task10_Colon/labelsTr -name '*.nii.gz' | wc -l
```

The image and label counts for the public labeled subset should match.

Each image should have a corresponding label with the same case identifier.

Example:

```text
imagesTr/colon_001.nii.gz
labelsTr/colon_001.nii.gz
```

---

## 1.5 How labels are used in this project

### Allowed

Ground-truth masks may be used to derive **evaluation-only** quantities after inference, including:

- whether an axial slice contains tumor;
- tumor slice interval;
- tumor bounding box;
- tumor centroid;
- pixel-level localization metrics;
- tumor-vs-background causal perturbation analysis.

### Forbidden

The mask must not be used to:

- crop the model input around the tumor;
- choose suspicious slices before MLLM inference;
- define candidate cells shown to the MLLM;
- optimize prompts;
- select thresholds using final test cases;
- train or tune any model component;
- generate task-specific reference features.

The correct workflow is:

```text
Raw CT
  ↓
Frozen MLLM inference
  ↓
Predicted anomaly scores / localization
  ↓
ONLY THEN compare against GT mask
```

---

## 1.6 Recommended preprocessing output

Keep the original NIfTI data untouched and write derived files separately:

```text
data/
├── raw/
│   └── MSD_Task10_Colon/
└── processed/
    └── MSD_Task10_Colon/
        ├── png_soft_tissue/
        │   └── colon_001/
        │       ├── slice_0000.png
        │       └── ...
        ├── metadata/
        └── evaluation_labels/
```

Suggested metadata fields:

```text
case_id
slice_index
spacing_x
spacing_y
spacing_z
width
height
tumor_present_for_evaluation_only
tumor_area_for_evaluation_only
```

Evaluation-only columns should never be included in MLLM prompts.

---

# 2. StageII-Colorectal-CT

## 2.1 What the dataset contains

StageII-Colorectal-CT is hosted by **The Cancer Imaging Archive (TCIA)**.

Official TCIA collection page:

```text
https://www.cancerimagingarchive.net/collection/stageii-colorectal-ct/
```

Persistent dataset DOI:

```text
https://doi.org/10.7937/p5k5-tg43
```

Key properties reported by TCIA:

- 230 patients;
- pathologically confirmed stage II colorectal cancer;
- 230 CT studies / 230 series;
- 13,850 CT images;
- approximately 7.31 GB;
- abdominal or pelvic contrast-enhanced CT before surgery;
- DICOM format.

Important limitation:

> The public TCIA collection does not provide an MSD-style primary colorectal tumor segmentation mask for routine quantitative localization evaluation.

Therefore this dataset is treated primarily as an **external zero-training validation cohort**.

---

## 2.2 Recommended download method: NBIA Data Retriever

TCIA commonly distributes imaging collections through an NBIA manifest (`.tcia`) that is opened using the **NBIA Data Retriever**.

### Step 1 — open the collection page

Go to:

```text
https://www.cancerimagingarchive.net/collection/stageii-colorectal-ct/
```

Use the collection download option to obtain the imaging manifest.

### Step 2 — install NBIA Data Retriever

Download the appropriate NBIA Data Retriever application from TCIA/NBIA instructions linked from the collection page.

### Step 3 — open the `.tcia` manifest

The retriever will download the DICOM studies while preserving patient/study/series structure.

Recommended destination:

```text
data/raw/StageII_Colorectal_CT/
```

Because TCIA download interfaces can change, the collection page and DOI should be treated as the canonical entry points rather than hard-coding a temporary manifest URL.

---

## 2.3 Alternative programmatic download

TCIA/NBIA offers programmatic access tools and APIs. For reproducible scripted acquisition, use the official NBIA/TCIA client workflow associated with the collection identifier.

Before implementing a custom downloader, verify the current TCIA API instructions because endpoint details and authentication requirements may change.

Canonical source:

```text
https://www.cancerimagingarchive.net/collection/stageii-colorectal-ct/
```

---

## 2.4 Recommended raw directory structure

Do not reorganize DICOM files destructively immediately after download.

Use:

```text
data/raw/StageII_Colorectal_CT/
├── <patient-id-1>/
│   └── <study>/
│       └── <series>/
│           ├── *.dcm
│           └── ...
├── <patient-id-2>/
└── ...
```

Create a separate processed NIfTI representation:

```text
data/processed/StageII_Colorectal_CT/
├── nifti/
│   ├── case_0001.nii.gz
│   └── ...
├── png_soft_tissue/
├── metadata/
└── reader_annotations/       # optional evaluation-only annotations
```

Keep a mapping table between anonymized project case IDs and the original TCIA patient/study/series identifiers.

---

## 2.5 DICOM-to-NIfTI conversion

Recommended tool:

```text
dcm2niix
```

Example:

```bash
dcm2niix \
  -z y \
  -f '%p_%s' \
  -o data/processed/StageII_Colorectal_CT/nifti \
  data/raw/StageII_Colorectal_CT/<case-folder>
```

Do not merge unrelated series automatically.

Before choosing a series for analysis, inspect:

- contrast phase;
- slice thickness;
- reconstruction kernel;
- anatomical coverage;
- number of slices;
- orientation;
- duplicated/localizer series.

The series-selection rule must be deterministic and documented before final analysis.

---

## 2.6 External validation without tumor masks

If no additional annotation is created, StageII can still be used for:

- zero-adaptation inference success rate;
- qualitative tumor-region proposal;
- causal anomaly score distribution;
- perturbation consistency;
- robustness across scanners/protocols;
- blinded radiologist confirmation of whether the top proposed region is clinically plausible.

Do **not** report Dice, IoU, pixel AUROC, or other dense localization metrics without a valid reference annotation.

---

## 2.7 Recommended minimal evaluation-only annotation

For a stronger publication, randomly pre-select approximately 50–100 StageII patients after the full method has been frozen.

A radiologist can provide only:

1. start/end axial slice containing the primary tumor;
2. one representative tumor slice;
3. one approximate 2D tumor bounding box.

This is intentionally much lighter than full 3D contouring.

These annotations must be stored separately:

```text
data/processed/StageII_Colorectal_CT/reader_annotations/
```

Suggested CSV schema:

```text
case_id
reader_id
tumor_slice_start
tumor_slice_end
representative_slice
bbox_xmin
bbox_ymin
bbox_xmax
bbox_ymax
```

They are used only for external evaluation and must never alter model prompts, thresholds, preprocessing, or selection rules.

---

# 3. Shared project data layout

Recommended final layout:

```text
training-free-prediction/
├── data/
│   ├── raw/
│   │   ├── MSD_Task10_Colon/
│   │   └── StageII_Colorectal_CT/
│   ├── processed/
│   │   ├── MSD_Task10_Colon/
│   │   └── StageII_Colorectal_CT/
│   └── metadata/
│       ├── msd_cases.csv
│       └── stageii_cases.csv
├── docs/
│   ├── research_plan.md
│   └── data_acquisition.md
└── ...
```

The entire `data/raw/`, `data/processed/`, and large output directories should be excluded from Git.

Suggested `.gitignore` entries:

```gitignore
data/raw/
data/processed/
*.nii
*.nii.gz
*.dcm
*.tar
*.zip
outputs/
```

Small metadata files containing no restricted patient information may be version controlled when their data licenses permit it.

---

# 4. Dataset use policy for the training-free claim

The following rule should appear in both the code and manuscript:

> **No ground-truth tumor annotation from either dataset is used during inference, proposal generation, prompt construction, score calculation, or parameter selection. Reference annotations are accessed only by the evaluation code after predictions have been saved.**

A robust implementation should physically separate:

```text
src/inference/
```

from:

```text
src/evaluation/
```

The inference pipeline should be able to run on a dataset directory containing **CT images only**.

This separation makes accidental label leakage much less likely.

---

# 5. Recommended acquisition order

## First

Download MSD Task10 Colon.

Reason:

- smaller and easier to process;
- NIfTI already available;
- expert tumor masks permit immediate quantitative testing;
- sufficient to determine whether the causal anomaly hypothesis is viable.

## Second

Download StageII-Colorectal-CT only after the MSD feasibility experiment succeeds.

Reason:

- external validation should be performed only after prompts and algorithmic choices are frozen;
- this prevents accidental external-test tuning;
- TCIA DICOM processing requires additional series-selection and conversion steps.

---

# 6. Minimum first experiment after download

After obtaining MSD, do **not** begin with the complete framework.

Run the smallest possible feasibility test:

1. export axial soft-tissue-window PNGs;
2. use the GT mask only to create hidden evaluation labels for `tumor-containing` vs `tumor-free` slices;
3. ask the frozen MLLM for an abnormality score on raw slices;
4. compare score distributions;
5. on tumor-containing slices, perturb the true tumor box **only inside the evaluation script** and compare its score drop with a matched background box;
6. separately test the practical method using MLLM-proposed cells instead of GT regions.

The central feasibility condition is:

```text
causal score drop at tumor region
>
causal score drop at matched non-tumor region
```

If this relation is robust at patient level, proceed to hierarchical localization and StageII external validation.

---

# 7. Citation / source notes

When publishing results, cite the original dataset providers rather than only repository mirrors.

## MSD Task10 Colon

- Antonelli M, et al. The Medical Segmentation Decathlon. *Nature Communications*. 2022.
- Official site: https://medicaldecathlon.com/

## StageII-Colorectal-CT

- TCIA collection: https://www.cancerimagingarchive.net/collection/stageii-colorectal-ct/
- DOI: https://doi.org/10.7937/p5k5-tg43
- Cite the original related StageII-Colorectal-CT publication specified on the TCIA collection page in the final manuscript.

Always re-check the current dataset license, citation instructions, and access conditions on the official collection pages before redistribution or publication.
