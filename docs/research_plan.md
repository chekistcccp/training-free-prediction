# CausalCRC-AD: Training-Free Causal Anomaly Detection and Localization for Colorectal Cancer CT

## 1. Project positioning

**Working title**

> **CausalCRC-AD: Training-Free Causal Anomaly Detection and Localization in Colorectal Cancer CT with Frozen Multimodal Large Language Models**

Alternative journal-style title:

> **Can Frozen Multimodal Large Language Models Localize Colorectal Cancer Without Training? A Causal Test-Time Anomaly Detection Framework**

This project studies whether a frozen multimodal large language model (MLLM), such as Qwen3.5-9B, can perform **colorectal cancer CT abnormality detection and localization without any task-specific model training**.

The central idea is to convert counterfactual image perturbation from a post-hoc explanation tool into the anomaly localization mechanism itself:

1. the frozen MLLM proposes suspicious slices/regions;
2. suspicious regions are locally neutralized at test time;
3. the change in the model's abnormality evidence is measured;
4. regions whose removal specifically decreases abnormality evidence receive a high causal anomaly score;
5. coarse-to-fine causal search produces an interpretable anomaly heatmap.

The study uses **only two colorectal cancer CT datasets**:

- Medical Segmentation Decathlon (MSD) Task10 Colon;
- StageII-Colorectal-CT from TCIA.

---

## 2. Core research question

> Can a frozen general-purpose multimodal large language model detect and localize primary colorectal cancer on CT through test-time causal intervention, without gradient-based training, fine-tuning, task-specific adapters, learned prompts, or a reference feature bank?

### Main hypotheses

**H1 — Training-free abnormal slice detection**  
A frozen MLLM can assign higher abnormality scores to CT slices containing primary colorectal tumors than to tumor-free slices from the same patients.

**H2 — Causal localization**  
Neutralizing a true tumor region should reduce the model's abnormality evidence substantially more than neutralizing a matched non-tumor region.

**H3 — Self-reported evidence is not always faithful**  
The region initially claimed by the MLLM to be abnormal will not always be causally responsible for the prediction; counterfactual verification can reject unfaithful proposals.

**H4 — Coarse-to-fine causal verification improves localization**  
Hierarchical causal refinement should improve lesion localization over direct zero-shot region proposal while requiring far fewer MLLM calls than exhaustive sliding-window occlusion.

**H5 — Cross-center transfer without adaptation**  
The complete inference procedure should transfer to StageII-Colorectal-CT without any parameter update or dataset-specific fitting.

---

## 3. Strict definition of “training-free”

The primary method must satisfy all of the following:

- no gradient computation;
- no fine-tuning of the MLLM;
- no LoRA / adapter / prompt tuning;
- no trained detection or segmentation head;
- no task-specific classifier;
- no training on MSD labels;
- no training on StageII-Colorectal-CT;
- no learned normal reference bank;
- no threshold fitted using ground-truth tumor annotations;
- no use of MSD tumor masks as model input;
- no use of StageII annotations for model adaptation.

Allowed operations:

- deterministic CT preprocessing;
- fixed CT windowing;
- deterministic resampling;
- frozen MLLM inference;
- structured prompts;
- multi-image / multi-slice inference;
- test-time perturbation;
- test-time self-consistency;
- deterministic aggregation of model outputs;
- ground-truth labels/masks **only for final evaluation**.

This strict definition is important because many “zero-shot anomaly detection” methods still learn prompts, adapters, or feature transformations on auxiliary datasets. This project intentionally studies the stricter setting of **no task-specific fitting at all**.

---

## 4. Dataset roles

### 4.1 MSD Task10 Colon — primary quantitative benchmark

The MSD Task10 Colon dataset contains **190 portal-venous phase CT scans** from patients undergoing resection of primary colon cancer. The tumor was manually segmented by an expert body radiologist. The public challenge structure includes labeled training data and unlabeled test data.

For this project, the labeled public cases are used only to construct evaluation targets after inference.

MSD provides the key quantitative benchmark because its 3D tumor masks allow objective evaluation of:

- tumor-containing vs tumor-free slices;
- top-k tumor slice retrieval;
- 2D tumor localization;
- causal anomaly map localization;
- tumor-vs-background causal score separation.

### 4.2 StageII-Colorectal-CT — external zero-training validation

StageII-Colorectal-CT contains **230 patients with pathologically confirmed stage II colorectal cancer**, with abdominal or pelvic contrast-enhanced CT acquired within 10 days before surgery. The current TCIA release contains 230 CT studies / 230 series / 13,850 DICOM images (about 7.31 GB).

The official public release does **not provide an MSD-style primary tumor segmentation mask**. Therefore it should not be used to report unverified Dice/IoU localization results.

Its role is:

- cross-center external validation;
- robustness to different scanners and acquisition protocols;
- confirmation that the inference framework remains usable without adaptation;
- qualitative / reader-confirmed tumor localization;
- optional minimal external annotation for a quantitative validation subset.

### Recommended optional StageII evaluation annotation

For a stronger Q1 journal submission, create a **small evaluation-only reader subset**, e.g. 50–100 randomly selected StageII cases. A radiologist only needs to record:

1. axial tumor-containing slice range;
2. one representative tumor slice;
3. a rough tumor bounding box on the representative slice.

Dense 3D contouring is not required.

These annotations are used **only after all prompts and method parameters are frozen** and never for training or tuning. This preserves the training-free claim while enabling external quantitative metrics such as top-k slice recall and pointing-game / box localization accuracy.

If no additional annotation is available, StageII remains an external qualitative and robustness cohort, while all quantitative localization claims must be based on MSD.

---

## 5. Task formulation

### Task A — slice-level anomaly detection

For each MSD volume, derive the evaluation label from the hidden tumor mask:

\[
y_z = \begin{cases}
1,& \sum_{x,y}M(x,y,z)>0\\
0,& \text{otherwise}
\end{cases}
\]

where `M` is the ground-truth tumor mask and `z` is the axial slice index.

The MLLM never receives `M`.

Goal:

> rank tumor-containing slices above tumor-free slices.

This is a particularly clean anomaly-detection setting because positive and negative slices come from the same cancer patient, reducing patient-level confounding.

### Task B — 2D causal anomaly localization

For a suspicious axial CT slice, identify the spatial region most causally responsible for the model's abnormality judgment.

Output:

- suspicious region proposal;
- causal anomaly score per region;
- coarse-to-fine anomaly heatmap;
- text description of the abnormal CT finding;
- faithfulness / verification status.

### Task C — volume-level tumor search

Aggregate slice-level abnormality scores to produce:

- top-1 / top-3 / top-5 suspicious slices;
- suspicious contiguous slice interval;
- case-level evidence summary.

### Task D — external zero-training validation on StageII

Without changing model weights, prompts, score definitions, preprocessing rules, or aggregation rules, apply the complete pipeline to StageII-Colorectal-CT.

---

## 6. Proposed method: CausalCRC-AD

### 6.1 Deterministic CT preprocessing

No learned preprocessing model is used.

Recommended default pipeline:

1. load volume;
2. orient consistently to axial RAS/LPS convention;
3. convert intensities to Hounsfield units when required;
4. use a pre-specified abdominal soft-tissue window (for example level 40 HU, width 400 HU);
5. resize exported 2D images to the MLLM input resolution while preserving aspect ratio;
6. optionally provide adjacent slices `(z-1, z, z+1)` as separate images for 2.5D context.

A multi-window variant may be tested as an ablation, but window settings must be fixed before evaluation and never optimized against the tumor masks.

### 6.2 Stage 1: training-free hierarchical slice screening

Running the MLLM independently on every CT slice may be expensive. Therefore use a hierarchical search.

#### Coarse screening

- partition a CT volume into contiguous blocks;
- sample representative slices from each block;
- compose them into a numbered contact sheet;
- ask the frozen MLLM to rank tiles by likelihood of containing a focal colorectal abnormality;
- retain top-k candidate blocks.

#### Fine screening

- expand candidate blocks to individual axial slices;
- score the candidate slices individually or in small neighboring groups;
- retain top-k suspicious slices for spatial analysis.

This reduces inference cost while preserving a training-free design.

### 6.3 Stage 2: MLLM abnormal region proposal

Overlay a fixed spatial grid, initially `4 × 4`, on the selected slice.

The MLLM receives a structured prompt requiring it to separate observation from diagnosis:

- NORMAL or ABNORMAL;
- top suspicious grid cell(s);
- visible CT evidence only;
- anatomical location;
- confidence / uncertainty;
- no access to ground-truth annotation.

Example output schema:

```json
{
  "status": "ABNORMAL",
  "candidate_cells": ["B3", "C3"],
  "finding": "focal irregular bowel-wall thickening with adjacent soft-tissue change",
  "anatomical_region": "left lower abdomen",
  "uncertainty": "moderate"
}
```

### 6.4 Stage 3: counterfactual causal intervention

For each candidate region `R_i`, construct counterfactual images in which the local visual evidence is neutralized.

Avoid relying only on a black rectangle because the mask artifact itself can change MLLM behavior.

Use several deterministic perturbations:

1. local Gaussian blur;
2. local mean / median intensity replacement;
3. surrounding-intensity interpolation or boundary-consistent fill.

No generative inpainting model is required for the primary implementation.

### 6.5 Global abnormality evidence

When local inference exposes token logits, define a logit-based abnormality score:

\[
S(I)=\log P(ABNORMAL|I)-\log P(NORMAL|I)
\]

This is preferred over asking the language model to invent a verbal percentage.

If token-level scores are unavailable in a deployment setting, use a fixed ordinal structured score as a secondary analysis, but the main local implementation should use logits where possible.

### 6.6 Causal Necessity Score

For candidate region `R_i`:

\[
N_i=S(I)-S(I^{-R_i})
\]

where `I^{-R_i}` is the counterfactual image after neutralizing region `R_i`.

Interpretation:

- large positive `N_i`: the removed region was important evidence for abnormality;
- near-zero `N_i`: the model's abnormality judgment did not depend on the proposed region;
- negative `N_i`: perturbing the region paradoxically increased abnormality evidence.

### 6.7 Matched negative-control region

For every candidate region, select one or more control regions of comparable area that were not proposed as suspicious.

Define:

\[
CAS(R_i)=\Delta_{candidate}-\operatorname{median}(\Delta_{control})
\]

with

\[
\Delta=S(I)-S(I^{-R})
\]

`CAS` is the **Causal Anomaly Score**.

This explicitly controls for the possibility that the model changes its prediction simply because any part of the image was altered.

### 6.8 Multi-perturbation consistency

For perturbation types `p = 1...P`:

\[
CAS_{multi}(R_i)=\frac{1}{P}\sum_p CAS_p(R_i)
\]

Also report perturbation agreement / variance as a reliability measure.

A truly causal lesion region should remain important across reasonable local-neutralization strategies.

### 6.9 Stage 4: coarse-to-fine spatial refinement

Only the highest-scoring coarse regions are subdivided.

Example:

- Level 1: whole image → `4 × 4` grid;
- retain top 1–3 cells;
- Level 2: each retained cell → `3 × 3` sub-grid;
- optionally one additional local refinement;
- interpolate region scores into a continuous anomaly heatmap.

This is substantially cheaper than exhaustive pixel-level or dense sliding-window perturbation.

### 6.10 Stage 5: multi-slice consistency

True colorectal tumors generally persist across adjacent axial slices.

For each candidate slice `z`, combine evidence from neighboring slices:

\[
S_{3D}(z)=\alpha S(z)+(1-\alpha)\operatorname{mean}[S(z-1),S(z+1)]
\]

For the strictest training-free version, use a fixed pre-specified `alpha` (e.g. 0.5) rather than fitting it.

An even simpler primary analysis can use a median score over `(z-1,z,z+1)`.

### 6.11 Final interpretable output

For each case, produce a structured report:

| Field | Output |
|---|---|
| Top suspicious slices | z1, z2, z3 |
| Proposed abnormal region | grid/sub-grid coordinates |
| CT finding | text observation |
| Raw abnormality score | S(I) |
| Causal anomaly score | CAS |
| Perturbation consistency | high / medium / low |
| Faithfulness status | verified / unverified |
| Final anomaly map | 2D heatmap |

This makes explanation faithfulness part of the method rather than an optional visualization.

---

## 7. Prompt design principles

Prompts should be fixed before reading the final evaluation results.

Key principles:

1. **observation before diagnosis** — ask for visible findings before a disease label;
2. **structured outputs** — require fixed JSON or table fields;
3. **no ground-truth hints** — never provide slice labels, tumor locations, masks, or patient-specific annotations;
4. **uncertainty allowed** — the model may answer uncertain rather than hallucinating a lesion;
5. **same prompt across datasets** — StageII must use the same prompt templates used for MSD;
6. **no few-shot examples from either study dataset** in the primary experiment.

---

## 8. Evaluation on MSD Colon

### 8.1 Slice-level anomaly detection

Primary metrics:

- AUROC;
- AUPRC;
- sensitivity at pre-specified specificity levels;
- tumor-slice ranking percentile;
- top-k tumor slice recall.

All confidence intervals should be patient-level bootstrap intervals rather than treating slices as independent samples.

### 8.2 Volume-level tumor search

Metrics:

- Top-1 tumor-containing slice hit rate;
- Top-3 hit rate;
- Top-5 hit rate;
- distance from highest-ranked slice to nearest tumor-containing slice;
- overlap between predicted suspicious slice interval and ground-truth tumor slice interval.

### 8.3 Spatial localization

Threshold-light / threshold-free metrics are preferred because no threshold should be tuned on the labels.

Recommended metrics:

- Pointing Game accuracy: whether the maximum anomaly-map point lies inside the tumor mask;
- pixel-level AUROC;
- AUPRO / region-overlap measures where practical;
- fraction of anomaly energy inside the tumor mask;
- center-distance between predicted maximum and tumor-mask centroid;
- bounding-box hit rate using the box derived from the GT mask.

Dice/IoU may be reported only with a **pre-registered, non-tuned** heatmap threshold (for example a fixed top percentile) and should not be the sole localization metric.

### 8.4 Causal faithfulness

Compare perturbation effects for:

- GT tumor region;
- MLLM-proposed region;
- matched non-tumor control region.

Key expected relation:

\[
\Delta_{tumor} > \Delta_{matched\ normal\ region}
\]

Report:

- median paired causal drop;
- proportion of cases satisfying the expected direction;
- effect size;
- paired confidence interval;
- perturbation-type consistency.

### 8.5 Self-reported evidence faithfulness

Measure whether the region the MLLM *claims* is important actually has high causal impact.

This enables a clinically meaningful distinction between:

- correct answer + faithful evidence;
- correct answer + unfaithful evidence;
- incorrect answer + plausible explanation;
- uncertain but causally coherent reasoning.

---

## 9. External validation on StageII-Colorectal-CT

### 9.1 Mandatory no-adaptation rule

Before StageII evaluation, freeze:

- all prompts;
- CT windowing;
- slice-search rules;
- number of candidate regions;
- perturbation types;
- aggregation rules;
- all score definitions.

No parameter may be adjusted after inspecting StageII outcomes.

### 9.2 Without new StageII annotations

Report:

- technical success rate;
- proportion of cases with localized suspicious evidence;
- distribution of causal anomaly scores;
- perturbation stability;
- qualitative case series covering different tumor locations and scanners;
- blinded expert rating of whether the top proposed region corresponds to the primary colorectal lesion, if reader review is available.

Do not report unsupported Dice/IoU.

### 9.3 Recommended minimal quantitative external validation

Randomly pre-select 50–100 StageII cases.

Have a radiologist record only:

- tumor-containing slice interval;
- representative tumor slice;
- approximate 2D bounding box.

Then report:

- Top-1 / Top-3 / Top-5 slice recall;
- pointing-game accuracy;
- predicted-max-point inside rough box;
- center distance;
- causal score difference between reader tumor box and matched control box.

Because these labels are created only for evaluation and never used to fit the method, the model remains training-free.

---

## 10. Baselines

The goal is not to compare against every supervised CRC detector. The main comparison should isolate the value of causal verification.

### Baseline B0 — direct zero-shot MLLM

Raw CT slice → frozen MLLM → NORMAL / ABNORMAL.

### Baseline B1 — direct MLLM region proposal

Raw CT slice + fixed grid → MLLM selects suspicious cell(s), without perturbation verification.

### Baseline B2 — exhaustive occlusion sensitivity

Use fixed grid perturbation everywhere without MLLM proposal. This tests whether hierarchical MLLM-guided search reduces computation while retaining localization quality.

### Baseline B3 — proposal + single perturbation

Candidate proposal + one perturbation type, no matched control.

### Proposed — full CausalCRC-AD

Candidate proposal + matched control + multi-perturbation CAS + hierarchical refinement + neighboring-slice consistency.

Optional secondary frozen vision-language baselines may be added only if they require no task-specific fitting.

---

## 11. Required ablation studies

Keep the ablation design simple and directly tied to the method.

1. Direct MLLM vs region proposal vs causal verification.
2. No matched control vs matched control.
3. Black mask vs blur vs intensity replacement vs multi-perturbation average.
4. `4 × 4` vs `6 × 6` coarse grid.
5. One-stage vs coarse-to-fine refinement.
6. Single slice vs three neighboring slices.
7. Single soft-tissue window vs fixed multi-window input.
8. Free-form prompt vs structured observation-first prompt.

No ablation setting should be chosen based on final test performance; define a development subset at the **patient level** if exploratory prompt engineering is necessary.

---

## 12. Statistical analysis

Recommended analysis:

- 1,000 or more patient-level bootstrap resamples for 95% confidence intervals;
- DeLong test for paired AUROC comparisons when appropriate;
- paired Wilcoxon signed-rank test for tumor-vs-control causal drops;
- effect sizes with confidence intervals;
- Holm correction for multiple primary paired comparisons;
- patient-level rather than slice-level resampling to avoid artificially narrow confidence intervals.

Pre-specify primary endpoints before final evaluation.

Suggested primary endpoints:

1. MSD slice-level AUROC;
2. MSD Top-3 tumor slice recall;
3. MSD Pointing Game localization accuracy;
4. paired tumor-vs-control CAS difference.

---

## 13. Efficiency analysis

Because MLLMs are expensive, computational efficiency should be a formal experiment.

Report:

- MLLM calls per CT volume;
- calls per localized slice;
- wall-clock inference time;
- GPU memory;
- number of visual tokens/images processed;
- localization performance vs number of causal interventions.

Compare hierarchical search against exhaustive grid occlusion.

A strong result would show that MLLM-guided proposals substantially reduce intervention count while maintaining or improving localization accuracy.

---

## 14. Failure-mode analysis

Manually categorize representative failures:

- bowel-wall thickening missed by the MLLM;
- stool / collapsed bowel mistaken for tumor;
- adjacent inflammatory change mistaken for tumor;
- lymph node or other mass mistaken for primary lesion;
- tumor proposal correct but causal verification weak;
- direct diagnosis correct but evidence region incorrect;
- CT slice thickness / contrast / artifact-related failure;
- MLLM textual hallucination of a finding not visible in the image.

These cases are valuable because the paper focuses on trustworthy reasoning, not only performance.

---

## 15. Recommended implementation sequence

### Phase 1 — fastest feasibility test

Use only the labeled MSD public cases.

Implement:

1. CT loading and soft-tissue rendering;
2. GT-derived slice labels for evaluation only;
3. frozen Qwen slice abnormality scoring;
4. direct `4 × 4` grid proposal;
5. candidate perturbation and matched-control perturbation;
6. CAS calculation;
7. Pointing Game and tumor-vs-control CAS analysis.

**Go/no-go criterion:** tumor-containing slices and tumor regions must show a meaningful score separation from patient-matched tumor-free slices/regions.

### Phase 2 — full coarse-to-fine method

Add:

- hierarchical slice search;
- multi-perturbation consistency;
- coarse-to-fine spatial refinement;
- neighboring-slice aggregation;
- full ablation table.

### Phase 3 — external validation

Run the frozen pipeline on StageII-Colorectal-CT.

Preferably add the minimal reader annotation subset for quantitative external validation.

---

## 16. Expected paper contributions

Keep the final paper to three main methodological contributions.

### Contribution 1 — Strict training-free CRC CT anomaly detection

A frozen MLLM performs abnormal slice discovery and lesion proposal without any task-specific optimization or normal reference bank.

### Contribution 2 — Counterfactual Causal Anomaly Score

Local perturbation plus matched negative controls verifies whether the proposed region is actually responsible for the model's abnormality judgment.

### Contribution 3 — Efficient hierarchical causal localization

MLLM-guided candidate selection plus coarse-to-fine refinement converts causal intervention into a practical tumor localization mechanism rather than an expensive post-hoc visualization.

A secondary scientific finding can be:

> diagnostic correctness and evidence faithfulness are not equivalent in frozen medical MLLMs.

---

## 17. Why the study may be publishable

The work should not be presented as “Qwen detects colon cancer.” That would be too application-oriented.

The stronger scientific framing is:

> **Can causal test-time intervention turn an untrained general-purpose MLLM into a self-verifying medical anomaly detector?**

The colorectal CT task is the clinical test bed, while the main methodological novelty is:

- strict no-training inference;
- causal rather than purely semantic localization;
- explanation faithfulness built into the detector;
- quantitative lesion localization using independent ground truth;
- external cross-center validation without adaptation.

A Q1 submission is plausible only if the final study shows convincing quantitative improvement over direct MLLM proposals and ordinary occlusion baselines, includes rigorous patient-level statistics, and avoids overstating StageII localization when no public GT annotation exists.

---

## 18. Reproducibility rules

- Fix random seeds for control-region sampling.
- Save every prompt and raw model response.
- Save model version / checkpoint hash.
- Save preprocessing parameters.
- Save every counterfactual image generated during evaluation.
- Separate exploratory development cases from final MSD evaluation cases at patient level.
- Never expose GT mask-derived information to the MLLM.
- Never use StageII to tune the method.
- Report failed / invalid model outputs rather than silently dropping them.

---

## 19. Suggested repository structure

```text
training-free-prediction/
├── docs/
│   ├── research_plan.md
│   └── data_acquisition.md
├── configs/
│   ├── model/
│   ├── prompts/
│   └── experiments/
├── data/
│   ├── raw/                 # gitignored
│   │   ├── MSD_Task10_Colon/
│   │   └── StageII_Colorectal_CT/
│   ├── processed/           # gitignored
│   └── metadata/
├── src/
│   ├── preprocessing/
│   ├── inference/
│   ├── perturbation/
│   ├── localization/
│   └── evaluation/
├── scripts/
└── outputs/                 # gitignored except small examples
```

---

## 20. Primary sources

### Medical Segmentation Decathlon

- Official website: https://medicaldecathlon.com/
- Official AWS download page: https://medicaldecathlon.com/dataaws/
- Task10 Colon direct archive used by MONAI: https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar
- MSD paper: Antonelli et al., *The Medical Segmentation Decathlon*, Nature Communications, 2022.

### StageII-Colorectal-CT

- TCIA collection: https://www.cancerimagingarchive.net/collection/stageii-colorectal-ct/
- Dataset DOI: https://doi.org/10.7937/p5k5-tg43
- Original related publication: Li et al., *International Journal of Cancer*, 2022.

---

## 21. Current recommended decision

Start with **MSD Task10 Colon only** until the following three observations are established:

1. tumor-containing slices have higher abnormality scores than tumor-free slices;
2. perturbing the true tumor region causes a larger abnormality-score drop than perturbing matched normal tissue;
3. causal verification improves localization over direct MLLM region proposals.

Only after these conditions are met should the frozen final pipeline be run on **StageII-Colorectal-CT** as the external validation cohort.
