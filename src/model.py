from __future__ import annotations
import math
from pathlib import Path
import torch

class Qwen35MedicalReasoner:
    def __init__(self,cfg):
        self.cfg=cfg; d=Path(cfg['local_dir'])
        if not d.exists(): raise FileNotFoundError(d)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        dm={'bfloat16':torch.bfloat16,'bf16':torch.bfloat16,'float16':torch.float16,'fp16':torch.float16,'float32':torch.float32,'fp32':torch.float32}
        self.dtype=dm[str(cfg.get('dtype','bfloat16')).lower()]
        from transformers import AutoProcessor
        try: from transformers import AutoModelForMultimodalLM as AutoMM
        except ImportError: from transformers import AutoModelForImageTextToText as AutoMM
        self.processor=AutoProcessor.from_pretrained(str(d),trust_remote_code=True,local_files_only=True)
        self.model=AutoMM.from_pretrained(str(d),trust_remote_code=True,local_files_only=True,torch_dtype=self.dtype,device_map={'':0},low_cpu_mem_usage=True,attn_implementation=cfg.get('attn_implementation','sdpa'))
        self.model.eval(); torch.set_grad_enabled(False); torch.backends.cuda.matmul.allow_tf32=True
        q=getattr(self.model.config,'quantization_config',None)
        if q not in (None,{}): raise RuntimeError(f'Quantized model detected: {q}')
        self.device=next(self.model.parameters()).device; self.tokenizer=self.processor.tokenizer
    def _messages(self,images,text):
        c=[{'type':'image','url':Path(p).resolve().as_uri()} for p in images]; c.append({'type':'text','text':text})
        return [{'role':'system','content':[{'type':'text','text':'You are a careful CT image reasoning assistant. Describe only visible evidence. Follow the requested output format exactly.'}]},{'role':'user','content':c}]
    def _templ(self,messages):
        kw=dict(add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
        if not self.cfg.get('enable_thinking',False): kw['enable_thinking']=False
        try: x=self.processor.apply_chat_template(messages,**kw)
        except TypeError: kw.pop('enable_thinking',None); x=self.processor.apply_chat_template(messages,**kw)
        return x.to(self.device)
    @torch.inference_mode()
    def generate(self,images,prompt):
        x=self._templ(self._messages(images,prompt)); n=x['input_ids'].shape[-1]
        y=self.model.generate(**x,max_new_tokens=int(self.cfg.get('max_new_tokens',192)),do_sample=False,use_cache=True)
        return self.processor.decode(y[0,n:],skip_special_tokens=True).strip()
    @torch.inference_mode()
    def sequence_logprob(self,images,prompt,candidate):
        x=self._templ(self._messages(images,prompt)); c=torch.tensor(self.tokenizer.encode(candidate,add_special_tokens=False),device=self.device).unsqueeze(0); n=x['input_ids'].shape[-1]
        z=dict(x); z['input_ids']=torch.cat([x['input_ids'],c],1)
        if 'attention_mask' in x: z['attention_mask']=torch.cat([x['attention_mask'],torch.ones_like(c,dtype=x['attention_mask'].dtype)],1)
        lp=torch.log_softmax(self.model(**z,use_cache=False).logits.float(),-1); vals=[lp[0,n+j-1,int(c[0,j])] for j in range(c.shape[1])]
        return float(torch.stack(vals).mean())
    @torch.inference_mode()
    def abnormality_score(self,image):
        p='Assess this single contrast-enhanced abdominal/pelvic CT image for a focal colorectal primary-tumor abnormality. Consider focal or asymmetric bowel-wall thickening, an irregular mass, luminal narrowing, or suspicious local soft-tissue extension. Ignore unrelated findings. Answer with exactly one word: NORMAL or ABNORMAL.'
        a=self.tokenizer.encode('ABNORMAL',add_special_tokens=False); n=self.tokenizer.encode('NORMAL',add_special_tokens=False)
        if len(a)==len(n)==1 and a[0]!=n[0]:
            x=self._templ(self._messages([image],p)); lp=torch.log_softmax(self.model(**x,use_cache=False).logits[0,-1].float(),-1); la=float(lp[a[0]]); ln=float(lp[n[0]])
        else: la=self.sequence_logprob([image],p,'ABNORMAL'); ln=self.sequence_logprob([image],p,'NORMAL')
        s=la-ln; pa=1/(1+math.exp(-max(min(s,60),-60)))
        return {'score':float(s),'p_abnormal':float(pa),'lp_abnormal':la,'lp_normal':ln}
    def clear_cache(self): torch.cuda.empty_cache()
