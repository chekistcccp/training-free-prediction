from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

def safe_auc(y_true,y_score):
    y=np.asarray(list(y_true)); s=np.asarray(list(y_score),float)
    return None if len(np.unique(y))<2 else float(roc_auc_score(y,s))

def safe_auprc(y_true,y_score):
    y=np.asarray(list(y_true)); s=np.asarray(list(y_score),float)
    return None if y.sum()==0 else float(average_precision_score(y,s))

def patient_bootstrap_auc(df,n_boot=1000,seed=20260901):
    patients=df.case_id.drop_duplicates().tolist(); rng=np.random.default_rng(seed); vals=[]
    for _ in range(n_boot):
        parts=[]
        for j,p in enumerate(rng.choice(patients,size=len(patients),replace=True)):
            x=df[df.case_id==p].copy(); x['_boot']=f'{p}_{j}'; parts.append(x)
        a=safe_auc(pd.concat(parts).label,pd.concat(parts).score)
        if a is not None: vals.append(a)
    point=safe_auc(df.label,df.score)
    return {'auc':point,'ci_low':float(np.percentile(vals,2.5)) if vals else None,'ci_high':float(np.percentile(vals,97.5)) if vals else None}

def topk_case_recall(df,ks=(1,3,5)):
    out={f'top{k}_recall':[] for k in ks}
    for _,g in df.groupby('case_id'):
        g=g.sort_values('score',ascending=False)
        for k in ks: out[f'top{k}_recall'].append(float(g.head(k).label.max()>0))
    return {k:float(np.mean(v)) for k,v in out.items()}

def pointing_game(heat,gt):
    if heat.size==0 or not np.isfinite(heat).any(): return False
    y,x=np.unravel_index(int(np.nanargmax(heat)),heat.shape); return bool(gt[y,x])

def energy_inside(heat,gt):
    h=np.clip(np.asarray(heat,float),0,None); t=h.sum(); return 0.0 if t<=0 else float(h[gt.astype(bool)].sum()/t)

def pixel_auc(heat,gt): return safe_auc(gt.reshape(-1).astype(np.uint8),heat.reshape(-1))

def summarize_localization(rows):
    if not rows:return {}
    d=pd.DataFrame(rows); v=pd.to_numeric(d.get('pixel_auc'),errors='coerce')
    return {'n':len(d),'pointing_accuracy':float(d.pointing.mean()),'mean_energy_inside':float(d.energy_inside.mean()),'mean_pixel_auc':float(v.mean()) if len(v) else None}
