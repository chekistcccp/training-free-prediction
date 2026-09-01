# Data directory

Medical data are intentionally excluded from git.

Place the manually downloaded datasets as follows:

```text
data/
├── MSD_Task10_Colon/
│   └── Task10_Colon/
│       ├── imagesTr/*.nii.gz
│       └── labelsTr/*.nii.gz
└── StageII-Colorectal-CT/
    └── <TCIA DICOM files in any nested hierarchy>
```

`Task10_Colon/` may be omitted if `imagesTr/` and `labelsTr/` are placed directly under `data/MSD_Task10_Colon/`.

The StageII loader recursively discovers DICOM files and groups CT files by `SeriesInstanceUID`.

Generated 512x512 CT renderings are stored in `data/processed/` and ignored by git.
