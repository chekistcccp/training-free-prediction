from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image

@dataclass
class MSDCase:
    case_id: str
    image_path: Path
    mask_path: Path

@dataclass
class StageIISeries:
    case_id: str
    series_uid: str
    files: list[Path]

def resolve_msd_root(root):
    root = Path(root)
    for c in [root, root/'Task10_Colon', root/'MSD_Task10_Colon'/'Task10_Colon']:
        if (c/'imagesTr').is_dir() and (c/'labelsTr').is_dir():
            return c
    raise FileNotFoundError(f"MSD Task10 Colon not found below {root}. Expected imagesTr/ and labelsTr/.")

def discover_msd_cases(root):
    root = resolve_msd_root(root); cases=[]
    for image_path in sorted((root/'imagesTr').glob('*.nii*')):
        name=image_path.name; mask_path=root/'labelsTr'/name
        if not mask_path.exists():
            stem=name.replace('.nii.gz','').replace('.nii',''); matches=list((root/'labelsTr').glob(stem+'.nii*'))
            if not matches: continue
            mask_path=matches[0]
        cases.append(MSDCase(name.replace('.nii.gz','').replace('.nii',''), image_path, mask_path))
    if not cases: raise RuntimeError(f"No labeled MSD cases found in {root}")
    return cases

def load_msd_case(case, canonicalize=True):
    img=nib.load(str(case.image_path)); msk=nib.load(str(case.mask_path))
    if canonicalize: img=nib.as_closest_canonical(img); msk=nib.as_closest_canonical(msk)
    image=np.asarray(img.dataobj,dtype=np.float32); mask=np.asarray(msk.dataobj)>0
    if image.ndim==4: image=image[...,0]
    if mask.ndim==4: mask=mask[...,0]
    if image.shape!=mask.shape: raise ValueError(f"shape mismatch {case.case_id}: {image.shape} vs {mask.shape}")
    return np.transpose(image,(2,1,0)), np.transpose(mask,(2,1,0))

def window_ct(x, level=40, width=400):
    lo,hi=level-width/2,level+width/2; x=np.clip(x,lo,hi); x=(x-lo)/max(hi-lo,1e-6)
    return np.round(x*255).astype(np.uint8)

def render_slice(x, level, width, size):
    img=Image.fromarray(window_ct(x,level,width),mode='L').convert('RGB')
    return img.resize((size,size),Image.Resampling.BILINEAR) if size>0 else img

def resize_mask(mask,size):
    return np.asarray(Image.fromarray((mask.astype(np.uint8)*255),mode='L').resize((size,size),Image.Resampling.NEAREST))>0

def save_rendered_slice(out_path,x,level,width,size):
    p=Path(out_path); p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists(): render_slice(x,level,width,size).save(p)
    return p

def _dicom_candidates(root):
    for p in Path(root).rglob('*'):
        if p.is_file() and p.stat().st_size>128: yield p

def discover_stageii_series(root):
    root=Path(root)
    if not root.exists(): raise FileNotFoundError(root)
    groups={}; patients={}
    for p in _dicom_candidates(root):
        try: ds=pydicom.dcmread(str(p),stop_before_pixels=True,specific_tags=['SeriesInstanceUID','PatientID','Modality'],force=True)
        except Exception: continue
        if str(getattr(ds,'Modality','')).upper()!='CT': continue
        uid=str(getattr(ds,'SeriesInstanceUID',''))
        if not uid: continue
        groups.setdefault(uid,[]).append(p); patients[uid]=str(getattr(ds,'PatientID','unknown')) or 'unknown'
    out=[StageIISeries(f"{patients[u]}__{u[-12:]}",u,fs) for u,fs in groups.items() if len(fs)>=8]
    if not out: raise RuntimeError(f"No CT DICOM series discovered below {root}")
    return sorted(out,key=lambda x:x.case_id)

def load_stageii_series(series):
    def key(p):
        try:
            ds=pydicom.dcmread(str(p),stop_before_pixels=True,specific_tags=['ImagePositionPatient','InstanceNumber'],force=True)
            ipp=getattr(ds,'ImagePositionPatient',None)
            if ipp is not None and len(ipp)>=3: return (0,float(ipp[2]))
            return (1,float(getattr(ds,'InstanceNumber',0)))
        except Exception: return (2,str(p))
    reader=sitk.ImageSeriesReader(); reader.SetFileNames([str(p) for p in sorted(series.files,key=key)])
    return sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)

def tumor_slice_labels(mask): return (mask.reshape(mask.shape[0],-1).sum(axis=1)>0).astype(np.uint8)
def largest_tumor_slice(mask): return int(np.argmax(mask.reshape(mask.shape[0],-1).sum(axis=1)))
