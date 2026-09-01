from __future__ import annotations
import cv2, numpy as np
from PIL import Image,ImageDraw,ImageFilter
from .utils import Box,grid_box,rc_to_cell

def draw_grid(image,grid,line_width=2):
    img=image.copy().convert('RGB'); d=ImageDraw.Draw(img); w,h=img.size
    for i in range(1,grid):
        x=round(w*i/grid); y=round(h*i/grid); d.line([(x,0),(x,h)],fill=(255,0,0),width=line_width); d.line([(0,y),(w,y)],fill=(255,0,0),width=line_width)
    for r in range(grid):
        for c in range(grid):
            b=grid_box(w,h,grid,r,c); lab=rc_to_cell(r,c); d.rectangle([b.x0+2,b.y0+2,b.x0+30,b.y0+22],fill=(0,0,0)); d.text((b.x0+5,b.y0+4),lab,fill=(255,255,0))
    return img

def perturb_image(image,box,kind,blur_radius=12,inpaint_radius=5):
    img=image.convert('RGB'); arr=np.asarray(img).copy(); x0,y0,x1,y1=box.x0,box.y0,box.x1,box.y1
    if kind=='blur':
        out=img.copy(); out.paste(img.crop((x0,y0,x1,y1)).filter(ImageFilter.GaussianBlur(blur_radius)),(x0,y0)); return out
    if kind=='mean_fill':
        pad=max(4,min(box.width,box.height)//12); xa,ya=max(0,x0-pad),max(0,y0-pad); xb,yb=min(arr.shape[1],x1+pad),min(arr.shape[0],y1+pad); fill=np.median(arr[ya:yb,xa:xb].reshape(-1,3),axis=0).astype(np.uint8); arr[y0:y1,x0:x1]=fill; return Image.fromarray(arr)
    if kind=='inpaint':
        bgr=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR); m=np.zeros(arr.shape[:2],np.uint8); m[y0:y1,x0:x1]=255; f=cv2.inpaint(bgr,m,float(inpaint_radius),cv2.INPAINT_TELEA); return Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    raise ValueError(kind)

def boxes_for_grid(image,grid,parent=None):
    w,h=image.size; return {rc_to_cell(r,c):grid_box(w,h,grid,r,c,parent) for r in range(grid) for c in range(grid)}

def box_mask(shape_hw,boxes,scores):
    h,w=shape_hw; heat=np.zeros((h,w),np.float32); weight=np.zeros((h,w),np.float32)
    for b,s in zip(boxes,scores): heat[b.y0:b.y1,b.x0:b.x1]+=float(s); weight[b.y0:b.y1,b.x0:b.x1]+=1
    v=weight>0; heat[v]/=weight[v]; return heat
