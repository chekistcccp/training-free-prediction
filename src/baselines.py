from __future__ import annotations
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from .data import discover_msd_cases,load_msd_case,largest_tumor_slice,resize_mask
from .utils import dump_json,ensure_dir


def run_exhaustive_occlusion_baseline(runner):
    """Fixed-grid exhaustive occlusion on the oracle largest-tumor MSD slice.

    Uses the first configured perturbation and all coarse grid cells. The tumor mask is
    used only after inference for localization evaluation.
    """
    rows=[]; od=ensure_dir(runner.out/'msd'/'exhaustive_occlusion'); size=int(runner.cfg['model']['image_size'])
    cases=runner._limit(discover_msd_cases(runner.paths['msd_root']))
    for c in tqdm(cases,desc='MSD exhaustive occlusion baseline'):
        jf=od/f'{c.case_id}.json'
        if runner.runtime.get('resume',True) and jf.exists(): rows.append(json.loads(jf.read_text())); continue
        vol,mask=load_msd_case(c,bool(runner.ct.get('canonicalize_nifti',True))); z=largest_tumor_slice(mask); p=runner._render('msd',c.case_id,z,vol[z]); gt=resize_mask(mask[z],size)
        out=runner._localize(c.case_id,p,f'exhaustive_{z}',exhaustive=True)
        # exhaustive=True makes every coarse cell a candidate; single_heatmap is standard
        # first-perturbation occlusion sensitivity without matched-control correction.
        m=runner._eval(out['single_heatmap'],gt)
        r={'case_id':c.case_id,'slice_idx':z,**m}; np.savez_compressed(od/f'{c.case_id}_heatmap.npz',heatmap=out['single_heatmap'].astype(np.float16),gt=gt.astype(np.uint8)); dump_json(jf,r); rows.append(r)
    d=pd.DataFrame(rows); sd=ensure_dir(runner.out/'summary'); d.to_csv(sd/'msd_exhaustive_occlusion.csv',index=False)
    if len(d):
        dump_json(sd/'msd_exhaustive_occlusion_metrics.json',{'n':len(d),'pointing_accuracy':float(d.pointing.mean()),'mean_energy_inside':float(d.energy_inside.mean()),'mean_pixel_auc':float(pd.to_numeric(d.pixel_auc,errors='coerce').mean())})
    return d
