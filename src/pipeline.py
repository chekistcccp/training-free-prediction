from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from .data import discover_msd_cases,discover_stageii_series,load_msd_case,load_stageii_series,save_rendered_slice,resize_mask,tumor_slice_labels,largest_tumor_slice
from .metrics import patient_bootstrap_auc,safe_auprc,topk_case_recall,pointing_game,energy_inside,pixel_auc
from .model import Qwen35MedicalReasoner
from .perturb import boxes_for_grid,box_mask,draw_grid,perturb_image
from .prompts import REGION_PROPOSAL
from .utils import ensure_dir,dump_json,parse_grid_cells,set_seed,stable_int

class ExperimentRunner:
    def __init__(self,cfg):
        self.cfg=cfg; set_seed(int(cfg['seed'])); self.paths=cfg['paths']; self.ct=cfg['ct']; self.loc=cfg['localization']; self.runtime=cfg['runtime']; self.out=ensure_dir(self.paths['output_root']); self.proc=ensure_dir(self.paths['processed_root']); self._model=None; self.calls=0
    @property
    def model(self):
        if self._model is None:self._model=Qwen35MedicalReasoner(self.cfg['model'])
        return self._model
    def _tick(self):
        self.calls+=1
        if self.calls%int(self.runtime.get('empty_cache_every',16))==0:self.model.clear_cache()
    def score(self,p):
        x=self.model.abnormality_score(p); self._tick(); return x
    def _render(self,dataset,case,z,arr):
        p=self.proc/dataset/case/f'slice_{z:04d}.png'; return save_rendered_slice(p,arr,float(self.ct['window_level']),float(self.ct['window_width']),int(self.cfg['model']['image_size']))
    def _limit(self,x,stage=False):
        v=self.runtime.get('stageii_max_cases' if stage else 'max_cases'); return x if v is None else x[:int(v)]
    def run_msd_slice_screening(self):
        rows=[]; od=ensure_dir(self.out/'msd'/'slice_scores'); stride=int(self.cfg['slice_screening'].get('stride',1))
        for c in tqdm(self._limit(discover_msd_cases(self.paths['msd_root'])),desc='MSD slice screening'):
            f=od/f'{c.case_id}.csv'
            if self.runtime.get('resume',True) and f.exists(): rows.append(pd.read_csv(f)); continue
            vol,mask=load_msd_case(c,bool(self.ct.get('canonicalize_nifti',True))); lab=tumor_slice_labels(mask); r=[]
            for z in range(0,len(vol),stride):
                p=self._render('msd',c.case_id,z,vol[z]); r.append({'case_id':c.case_id,'slice_idx':z,'label':int(lab[z]),**self.score(p)})
            d=pd.DataFrame(r); d.to_csv(f,index=False); rows.append(d)
        d=pd.concat(rows,ignore_index=True); sd=ensure_dir(self.out/'summary'); d.to_csv(sd/'msd_slice_scores.csv',index=False)
        dump_json(sd/'msd_slice_metrics.json',{'slice_auc':patient_bootstrap_auc(d,1000,int(self.cfg['seed'])),'slice_auprc':safe_auprc(d.label,d.score),**topk_case_recall(d,tuple(self.cfg['slice_screening']['top_k'])),'n_cases':int(d.case_id.nunique()),'n_slices':len(d),'n_positive_slices':int(d.label.sum())})
        return d
    def _proposal(self,case,img,tag):
        g=int(self.loc['coarse_grid']); k=int(self.loc['proposal_top_k']); td=ensure_dir(self.out/'tmp'/'grid'); p=td/f'{case}_{tag}.png'; draw_grid(img,g).save(p); text=self.model.generate([p],REGION_PROPOSAL.format(grid=g,top_k=k)); self._tick(); cells=parse_grid_cells(text,g)[:k]; return cells,text,bool(cells)
    def _cf(self,case,tag,img,box,kind,suffix):
        td=ensure_dir(self.out/'tmp'/'counterfactuals'/case); p=td/f'{tag}_{suffix}_{kind}.png'; perturb_image(img,box,kind,int(self.loc.get('blur_radius',12)),int(self.loc.get('inpaint_radius',5))).save(p); s=self.score(p)['score']
        if not self.runtime.get('save_counterfactuals',False): p.unlink(missing_ok=True)
        return s
    def _localize(self,case,p,tag,exhaustive=False):
        img=Image.open(p).convert('RGB'); g=int(self.loc['coarse_grid']); boxes=boxes_for_grid(img,g); allc=list(boxes); base=self.score(p)['score']; proposed,text,ok=self._proposal(case,img,tag); cand=allc if exhaustive or not proposed else proposed
        pool=[c for c in allc if c not in cand]; rng=np.random.default_rng(stable_int(case+tag)+int(self.cfg['seed'])); controls=list(rng.choice(pool,size=min(int(self.loc.get('control_count',3)),len(pool)),replace=False)) if pool else []
        perts=list(self.loc['perturbations']); ctrl={q:[] for q in perts}
        for q in perts:
            for c in controls: ctrl[q].append(base-self._cf(case,tag,img,boxes[c],q,'ctrl_'+c))
        rs=[]
        for c in cand:
            drops={}; cas={}
            for q in perts:
                d=base-self._cf(case,tag,img,boxes[c],q,'cand_'+c); drops[q]=float(d); cas[q]=float(d-(np.median(ctrl[q]) if ctrl[q] else 0))
            rs.append({'cell':c,'drop_mean':float(np.mean(list(drops.values()))),'cas':float(np.mean(list(cas.values()))),'drops':drops,'cas_by_perturbation':cas})
        rs.sort(key=lambda x:x['cas'],reverse=True); bb=[boxes[x['cell']] for x in rs]; full_scores=[max(0,x['cas']) for x in rs]; single=[max(0,x['drops'][perts[0]]) for x in rs]; noc=[max(0,x['drop_mean']) for x in rs]
        heat_single=box_mask((img.height,img.width),bb,single); heat_noc=box_mask((img.height,img.width),bb,noc)
        for coarse in rs[:int(self.loc.get('refine_top_k',1))]:
            sub=boxes_for_grid(img,int(self.loc.get('refine_grid',3)),boxes[coarse['cell']])
            for sc,b in sub.items():
                dd=[base-self._cf(case,tag,img,b,q,f'refine_{coarse["cell"]}_{sc}') for q in perts]; bb.append(b); full_scores.append(max(0,coarse['cas'])+max(0,float(np.mean(dd))))
        return {'base_score':base,'proposal_cells':proposed,'proposal_text':text,'proposal_parse_ok':ok,'fallback_exhaustive':not proposed and not exhaustive,'control_cells':controls,'cell_scores':rs,'heatmap':box_mask((img.height,img.width),bb,full_scores),'single_heatmap':heat_single,'nocontrol_heatmap':heat_noc}
    def _proposal_heat(self,case,p,tag):
        img=Image.open(p).convert('RGB'); cells,text,ok=self._proposal(case,img,tag); bd=boxes_for_grid(img,int(self.loc['coarse_grid'])); return cells,text,ok,box_mask((img.height,img.width),[bd[c] for c in cells],[1/(i+1) for i,c in enumerate(cells)])
    def _eval(self,h,gt): return {'pointing':int(pointing_game(h,gt)),'energy_inside':energy_inside(h,gt),'pixel_auc':pixel_auc(h,gt)}
    def _fig(self,p,h,out,title):
        out.parent.mkdir(parents=True,exist_ok=True); fig=plt.figure(figsize=(6,6)); plt.imshow(np.asarray(Image.open(p).convert('L')),cmap='gray');
        if np.max(h)>0: plt.imshow(h,alpha=.45)
        plt.title(title); plt.axis('off'); fig.savefig(out,dpi=140,bbox_inches='tight'); plt.close(fig)
    def run_msd_localization(self,scores,mode):
        rows=[]; od=ensure_dir(self.out/'msd'/f'localization_{mode}'); size=int(self.cfg['model']['image_size'])
        for c in tqdm(self._limit(discover_msd_cases(self.paths['msd_root'])),desc=f'MSD localization {mode}'):
            jf=od/f'{c.case_id}.json'
            if self.runtime.get('resume',True) and jf.exists(): rows.append(json.loads(jf.read_text())); continue
            vol,mask=load_msd_case(c,bool(self.ct.get('canonicalize_nifti',True))); z=largest_tumor_slice(mask) if mode=='oracle' else int(scores[scores.case_id==c.case_id].sort_values('score',ascending=False).iloc[0].slice_idx); p=self._render('msd',c.case_id,z,vol[z]); gt=resize_mask(mask[z],size)
            cells,text,ok,ph=self._proposal_heat(c.case_id,p,f'{mode}_proposal_{z}'); ca=self._localize(c.case_id,p,f'{mode}_causal_{z}')
            r={'case_id':c.case_id,'mode':mode,'slice_idx':z,'slice_has_tumor':int(gt.any()),'proposal_parse_ok':int(ok),'proposal_cells':cells,'proposal_text':text,'causal_base_score':ca['base_score'],'causal_cells':ca['cell_scores'],'control_cells':ca['control_cells'],'fallback_exhaustive':ca['fallback_exhaustive']}
            for pre,h in [('proposal_',ph),('single_',ca['single_heatmap']),('nocontrol_',ca['nocontrol_heatmap']),('',ca['heatmap'])]: r.update({pre+k:v for k,v in self._eval(h,gt).items()})
            np.savez_compressed(od/f'{c.case_id}_heatmaps.npz',causal=ca['heatmap'].astype(np.float16),proposal=ph.astype(np.float16),single=ca['single_heatmap'].astype(np.float16),nocontrol=ca['nocontrol_heatmap'].astype(np.float16),gt=gt.astype(np.uint8)); self._fig(p,ca['heatmap'],od/'figures'/f'{c.case_id}.png',f'{c.case_id} z={z} {mode}'); dump_json(jf,r); rows.append(r)
        d=pd.DataFrame(rows); sd=ensure_dir(self.out/'summary'); d.to_csv(sd/f'msd_localization_{mode}.csv',index=False)
        def sm(pre): return {'pointing_accuracy':float(d[pre+'pointing'].mean()),'mean_energy_inside':float(d[pre+'energy_inside'].mean()),'mean_pixel_auc':float(pd.to_numeric(d[pre+'pixel_auc'],errors='coerce').mean())}
        dump_json(sd/f'msd_localization_{mode}_metrics.json',{'full_method':sm(''),'proposal_only':sm('proposal_'),'single_perturbation_no_control':sm('single_'),'multi_perturbation_no_control':sm('nocontrol_')}); return d
    def run_stageii_external(self):
        rows=[]; od=ensure_dir(self.out/'stageii'); stride=int(self.cfg['slice_screening'].get('stride',1))
        for c in tqdm(self._limit(discover_stageii_series(self.paths['stageii_root']),True),desc='StageII external'):
            jf=od/f'{c.case_id}.json'
            if self.runtime.get('resume',True) and jf.exists(): rows.append(json.loads(jf.read_text())); continue
            vol=load_stageii_series(c); sr=[]
            for z in range(0,len(vol),stride): p=self._render('stageii',c.case_id,z,vol[z]); sr.append({'slice_idx':z,**self.score(p)})
            d=pd.DataFrame(sr).sort_values('score',ascending=False); top=d.head(5).to_dict('records'); z=int(top[0]['slice_idx']); p=self.proc/'stageii'/c.case_id/f'slice_{z:04d}.png'; ca=self._localize(c.case_id,p,f'external_{z}'); r={'case_id':c.case_id,'series_uid':c.series_uid,'n_slices':len(vol),'top_slices':top,'top_slice':z,'proposal_cells':ca['proposal_cells'],'proposal_text':ca['proposal_text'],'proposal_parse_ok':int(ca['proposal_parse_ok']),'fallback_exhaustive':ca['fallback_exhaustive'],'causal_cells':ca['cell_scores'],'control_cells':ca['control_cells'],'max_cas':float(max([x['cas'] for x in ca['cell_scores']],default=0))}; np.savez_compressed(od/f'{c.case_id}_heatmap.npz',heatmap=ca['heatmap'].astype(np.float16)); self._fig(p,ca['heatmap'],od/'figures'/f'{c.case_id}.png',f'StageII {c.case_id} z={z}'); dump_json(jf,r); rows.append(r)
        d=pd.DataFrame(rows); sd=ensure_dir(self.out/'summary'); d.to_csv(sd/'stageii_external_results.csv',index=False); dump_json(sd/'stageii_external_summary.json',{'n_series':len(d),'proposal_parse_rate':float(d.proposal_parse_ok.mean()),'median_max_cas':float(d.max_cas.median()),'note':'No public primary-tumor mask assumed; no Dice/IoU claim.'}); return d
