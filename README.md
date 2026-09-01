# CausalCRC-AD

Training-free causal anomaly detection and localization for **colorectal cancer CT** using the exact **Qwen/Qwen3.5-9B** multimodal model.

The repository is intentionally strict:

- Qwen3.5-9B only;
- model weights downloaded from ModelScope;
- BF16 by default (FP16/FP32 allowed, **no quantization**);
- no LoRA, adapter, prompt tuning, trained classifier, trained detector, or reference feature bank;
- MSD tumor masks are used only for evaluation;
- StageII-Colorectal-CT is used only as a zero-adaptation external cohort.

## Hardware target

Primary target: **one NVIDIA RTX 4090-class 48 GB GPU under WSL2/Linux**.

The official Qwen3.5-9B checkpoint is roughly 19 GB of weights. This implementation uses Transformers native inference with PyTorch SDPA rather than starting a separate serving engine, leaving headroom for the vision encoder, activations, and CT images on a 48 GB card.

## 1. Put data in `data/`

Do not commit medical data.

```text
data/
├── MSD_Task10_Colon/
│   └── Task10_Colon/
│       ├── imagesTr/
│       └── labelsTr/
└── StageII-Colorectal-CT/
    └── ... TCIA DICOM hierarchy ...
```

The MSD loader also accepts `data/MSD_Task10_Colon/imagesTr` directly. See `docs/data_acquisition.md`.

## 2. Run everything

From WSL/Linux:

```bash
chmod +x run.sh
./run.sh
```

`run.sh` will create `.venv`, install dependencies, verify CUDA/GPU/data layout, run unit tests, download **Qwen/Qwen3.5-9B** from ModelScope if needed, run all enabled MSD experiments, run StageII external inference, and aggregate results under `outputs/summary/`.

All expensive steps are cached per case. Re-running `./run.sh` resumes instead of restarting completed cases.

To force dependency reinstallation:

```bash
FORCE_INSTALL=1 ./run.sh
```

## 3. Experiments executed

### MSD Task10 Colon

- slice-level NORMAL/ABNORMAL scoring using Qwen token likelihoods;
- patient-level bootstrap AUROC and AUPRC;
- Top-1/3/5 tumor-slice retrieval;
- oracle-slice spatial localization (component analysis);
- end-to-end localization using the model's top-ranked slice;
- Qwen grid proposal baseline;
- exhaustive occlusion baseline;
- proposal + single perturbation baseline;
- proposal + multi-perturbation without matched controls;
- matched-control Causal Anomaly Score (CAS);
- coarse-to-fine causal refinement;
- Pointing Game, energy-inside-mask and pixel AUROC.

### StageII-Colorectal-CT

- identical frozen slice scoring;
- top suspicious slice retrieval;
- identical causal region proposal / verification;
- qualitative heatmaps and structured outputs;
- no Dice/IoU is claimed unless an evaluation-only reader annotation is added separately.

## 4. Abnormality score

```text
score = log P(ABNORMAL | image,prompt) - log P(NORMAL | image,prompt)
```

If each label is one token, one forward pass is used. Otherwise the code falls back to teacher-forced sequence likelihood. The model is not asked to invent a verbal confidence percentage.

## 5. Causal anomaly localization

For candidate region `R`:

```text
Delta(R) = S(original) - S(region-neutralized)
CAS(R)   = Delta(candidate) - median Delta(matched controls)
```

Default deterministic perturbations are Gaussian blur, local context fill, and OpenCV Telea inpainting. The highest-scoring coarse 4x4 region is refined with a 3x3 sub-grid.

## 6. Memory / precision policy

The loader never enables 4-bit/8-bit mode or a quantization config and explicitly checks for quantization metadata. Default dtype is BF16. `run.sh` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and the pipeline periodically clears unused CUDA cache.

## 7. Smoke run

For a short test edit `runtime.max_cases` and `runtime.stageii_max_cases` in `configs/experiment.yaml`. Restore both to `null` for the full paper experiment.

## 8. Research protocol

See `docs/research_plan.md`. Ground-truth masks/labels must never be inserted into prompts or used to tune inference thresholds.
